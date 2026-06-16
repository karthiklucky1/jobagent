"""FAISS-backed semantic matcher.

- SentenceTransformer all-MiniLM-L6-v2 produces 384-dim embeddings
- FAISS IndexFlatIP for cosine sim (vectors are L2-normalized)
- Index persists to disk; rebuilt only when stale

Resume goes through the same encoder so queries and corpus are in the same space.
"""
from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import Job
from app.qa_store.resolver import QAResolver
from app.matching.filters.constants import NON_US_LOCATIONS

log = logging.getLogger(__name__)

# Initialize canonical QA Resolver
qa_resolver = QAResolver()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384


def _chunk_resume(resume_text: str) -> List[str]:
    """Split markdown resume by headers and append a target skills profile chunk."""
    raw_chunks = []
    current_chunk = []
    for line in resume_text.split("\n"):
        if line.startswith("## ") or line.startswith("# "):
            if current_chunk:
                raw_chunks.append("\n".join(current_chunk).strip())
            current_chunk = [line]
        else:
            current_chunk.append(line)
    if current_chunk:
        raw_chunks.append("\n".join(current_chunk).strip())
        
    # Keep only non-empty, reasonably-sized chunks
    chunks = [c for c in raw_chunks if len(c.strip()) > 50]
    
    # Generate search profile dynamically from Q&A store fields
    data = qa_resolver.data
    identity = data.get("identity", {})
    bg = data.get("background", {})
    pref = data.get("preferences", {})

    roles = "AI/ML Engineer, NLP Engineer, MLOps/Platform Engineer, or Backend Python Developer"
    tech = bg.get("tech_stack", "Python, PyTorch, TensorFlow, Scikit-learn, XGBoost, NLP, Generative AI, LLMs, RAG, LangChain, Multi-Agent Systems, Spark, PySpark, MLflow, Kubeflow, Vertex AI, BigQuery, Kafka, Airflow, AWS, GCP, Docker, Kubernetes, FastAPI, FAISS, PostgreSQL, MongoDB")
    loc = identity.get("location", "Cincinnati, OH")
    arrangement = pref.get("work_arrangement", "Open to remote, hybrid, or onsite")

    summary_chunk = (
        f"Role Target: {roles}.\n"
        f"Preferred Location: {loc} ({arrangement}).\n"
        f"Key Technologies: {tech}."
    )
    chunks.append(summary_chunk)
    return chunks


class Matcher:
    def __init__(self):
        log.info("Loading embedding model %s …", MODEL_NAME)
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        log.info("Loading cross-encoder model mixedbread-ai/mxbai-rerank-xsmall-v1 …")
        self.cross_encoder = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1", device="cpu")
        self.index_path: Path = settings.faiss_index_path
        self.id_map_path: Path = self.index_path.with_suffix(".ids.npy")
        self.index: "faiss.Index" | None = None
        self.job_ids: np.ndarray | None = None  # index position -> Job.id

    # ---------- embeddings ----------

    def encode(self, texts: List[str]) -> np.ndarray:
        """Returns L2-normalized vectors (so inner product == cosine)."""
        import faiss
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embs)
        return embs.astype("float32")

    @staticmethod
    def _job_text(job: Job) -> str:
        """Document text we embed for each job. Title weighted heavily."""
        return f"{job.title}\n{job.title}\n{job.company} | {job.location}\n\n{job.description[:4000]}"

    # ---------- index lifecycle ----------

    def rebuild(self, user_id: str | None = None) -> int:
        """Incrementally update the FAISS index with any unindexed jobs in the DB."""
        import faiss

        # Load existing index if available
        existing_index = None
        existing_ids = set()

        if self.index_path.exists() and self.id_map_path.exists():
            try:
                existing_index = faiss.read_index(str(self.index_path))
                existing_ids_arr = np.load(self.id_map_path)
                existing_ids = set(existing_ids_arr.tolist())
                self.index = existing_index
                self.job_ids = existing_ids_arr
                log.info("Loaded existing FAISS index with %d vectors.", len(existing_ids))
            except Exception as e:
                log.warning("Failed to load existing FAISS index, rebuilding from scratch: %s", e)
                existing_index = None

        with get_session() as session:
            # Only index open jobs — closed/purged jobs must never re-enter matching.
            q = select(Job).where(Job.is_closed == False)
            if user_id:
                q = q.where(Job.user_id == user_id)
            all_jobs = session.exec(q).all()

        if not all_jobs:
            log.warning("No jobs in DB to index.")
            return 0
            
        # Find new and updated jobs
        new_jobs = [j for j in all_jobs if j.id not in existing_ids]
        updated_jobs = [j for j in all_jobs if j.id in existing_ids and j.embedding_id is None]
        
        # If there are updated jobs, we must do a full rebuild to clean up stale vectors
        force_rebuild = len(updated_jobs) > 0
        
        if not new_jobs and not force_rebuild:
            log.info("All %d jobs are already indexed. No update needed.", len(all_jobs))
            return len(all_jobs)
            
        if force_rebuild or existing_index is None or self.job_ids is None:
            # Build from scratch
            log.info("Building new FAISS index from scratch with %d vectors (forced rebuild=%s)...", len(all_jobs), force_rebuild)
            texts = [self._job_text(j) for j in all_jobs]
            embs = self.encode(texts)
            index = faiss.IndexFlatIP(DIM)
            index.add(embs)
            ids = np.array([j.id for j in all_jobs], dtype="int64")
            np.save(self.id_map_path, ids)
            faiss.write_index(index, str(self.index_path))
            self.index = index
            self.job_ids = ids
            
            # Save embedding_ids back to DB
            with get_session() as session:
                for idx, j in enumerate(all_jobs):
                    db_job = session.get(Job, j.id)
                    if db_job:
                        db_job.embedding_id = idx
                        session.add(db_job)
                session.commit()
                
            log.info("FAISS index built from scratch: %d vectors", len(all_jobs))
        else:
            # Incremental append
            log.info("Indexing %d new jobs incrementally...", len(new_jobs))
            new_texts = [self._job_text(j) for j in new_jobs]
            new_embs = self.encode(new_texts)
            
            start_idx = len(self.job_ids)
            existing_index.add(new_embs)
            new_ids_arr = np.array([j.id for j in new_jobs], dtype="int64")
            self.job_ids = np.concatenate([self.job_ids, new_ids_arr])
            faiss.write_index(existing_index, str(self.index_path))
            np.save(self.id_map_path, self.job_ids)
            
            # Save embedding_ids back to DB
            with get_session() as session:
                for idx, j in enumerate(new_jobs):
                    db_job = session.get(Job, j.id)
                    if db_job:
                        db_job.embedding_id = start_idx + idx
                        session.add(db_job)
                session.commit()
                
            log.info("FAISS index updated. Total vectors: %d", len(self.job_ids))
            
        return len(self.job_ids)

    def load(self) -> None:
        import faiss
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index missing at {self.index_path}; run rebuild() first.")
        self.index = faiss.read_index(str(self.index_path))
        self.job_ids = np.load(self.id_map_path)
        log.info("FAISS index loaded: %d vectors", self.index.ntotal)

    # ---------- search ----------

    def search_for_resume(self, resume_text: str, k: int = 30, user_id: str | None = None) -> List[Tuple[int, float]]:
        """Hybrid search with RRF (Max-Similarity chunked query) + local Cross-Encoder reranking.
        
        Returns [(job_id, cross_encoder_score)] sorted desc.
        """
        # Load embedding index if needed
        if self.index is None or self.job_ids is None:
            self.load()

        with get_session() as session:
            # Exclude closed/purged jobs from candidate retrieval.
            q = select(Job).where(Job.is_closed == False)
            if user_id:
                q = q.where(Job.user_id == user_id)
            jobs = session.exec(q).all()

        if not jobs:
            return []

        # Filter out jobs outside the US or with foreign locations (uses shared constants)
        non_us_locations = NON_US_LOCATIONS
        
        filtered_jobs = []
        for j in jobs:
            loc_low = (j.location or "").lower()
            title_low = j.title.lower()
            
            is_outside = False
            # Check location field
            if loc_low:
                for loc in non_us_locations:
                    if loc in loc_low:
                        is_outside = True
                        break
            else:
                # Check title context
                for loc in non_us_locations:
                    # Look for word boundaries e.g. "Korea" or "(Korea)" or "Seoul"
                    pattern = rf"\b{loc}\b"
                    if re.search(pattern, title_low) or (f"({loc})" in title_low):
                        is_outside = True
                        break
            
            if not is_outside:
                filtered_jobs.append(j)
                
        jobs = filtered_jobs

        if not jobs:
            return []

        # Split resume into chunks to prevent vector dilution / sequence truncation
        chunks = _chunk_resume(resume_text)
        log.info("Resume split into %d query chunks for matching.", len(chunks))

        # Build a focused profile string for cross-encoder (last chunk = summary profile)
        profile_chunk = chunks[-1] if chunks else resume_text[:2000]

        # 1. Lexical Search (BM25) with Max-Similarity
        def _tokenize(text: str) -> List[str]:
            return text.lower().split()

        tokenized_corpus = [_tokenize(self._job_text(j)) for j in jobs]
        bm25 = BM25Okapi(tokenized_corpus)

        job_max_bm25 = {j.id: -999999.0 for j in jobs}
        for chunk in chunks:
            tokenized_query = _tokenize(chunk)
            scores = bm25.get_scores(tokenized_query)
            for idx, score in enumerate(scores):
                jid = jobs[idx].id
                if score > job_max_bm25[jid]:
                    job_max_bm25[jid] = score

        bm25_ranking = sorted(jobs, key=lambda j: job_max_bm25[j.id], reverse=True)
        bm25_ranks = {j.id: rank for rank, j in enumerate(bm25_ranking)}

        # 2. Semantic Search (FAISS) with Max-Similarity
        chunk_embs = self.encode(chunks)
        faiss_scores, faiss_idxs = self.index.search(chunk_embs, len(jobs))

        job_max_faiss = {j.id: -1.0 for j in jobs}
        for chunk_idx in range(len(chunks)):
            scores = faiss_scores[chunk_idx]
            idxs = faiss_idxs[chunk_idx]
            for score, idx in zip(scores, idxs):
                if idx >= 0:
                    jid = int(self.job_ids[idx])
                    if jid in job_max_faiss and score > job_max_faiss[jid]:
                        job_max_faiss[jid] = score

        faiss_ranking = sorted(jobs, key=lambda j: job_max_faiss[j.id], reverse=True)
        faiss_ranks = {j.id: rank for rank, j in enumerate(faiss_ranking)}

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: List[Tuple[Job, float]] = []
        for j in jobs:
            b_rank = bm25_ranks.get(j.id, len(jobs))
            f_rank = faiss_ranks.get(j.id, len(jobs))
            rrf_score = 1.0 / (60.0 + b_rank) + 1.0 / (60.0 + f_rank)
            rrf_scores.append((j, rrf_score))

        # Send top max(k, 200) candidates through Cross-Encoder (was hardcoded to 50)
        ce_batch_size = max(k, 200)
        rrf_ranking = sorted(rrf_scores, key=lambda x: x[1], reverse=True)
        top_candidates = rrf_ranking[:ce_batch_size]
        log.info("Sending %d candidates to cross-encoder (from %d total)", len(top_candidates), len(jobs))

        # 4. Cross-Encoder Reranking — use profile chunk for sharper signal
        pairs = [(profile_chunk, self._job_text(j)) for j, _ in top_candidates]
        logits = self.cross_encoder.predict(pairs, show_progress_bar=True)

        # Sigmoid function to normalize logits to a 0-1 probability score
        scores_norm = 1.0 / (1.0 + np.exp(-logits))

        final_scores = []
        for (job, _), score in zip(top_candidates, scores_norm):
            final_scores.append((job.id, float(score)))

        # Return top k sorted by Cross-Encoder score descending
        final_ranking = sorted(final_scores, key=lambda x: x[1], reverse=True)
        log.info("Cross-encoder top scores: %s", [(s[1]) for s in final_ranking[:5]])
        return final_ranking[:k]
