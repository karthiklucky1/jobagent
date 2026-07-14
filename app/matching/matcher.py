"""FAISS-backed semantic matcher.

- SentenceTransformer all-MiniLM-L6-v2 produces 384-dim embeddings
- FAISS IndexFlatIP for cosine sim (vectors are L2-normalized)
- Index persists to disk; rebuilt only when stale

Resume goes through the same encoder so queries and corpus are in the same space.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import Job
from app.qa_store.resolver import QAResolver
from app.common.geo import detect_country, norm_country

log = logging.getLogger(__name__)

# Initialize canonical QA Resolver
qa_resolver = QAResolver()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384

# Max jobs indexed into FAISS per user. Bounds the from-scratch rebuild so a
# bloated pool can't turn it into a multi-minute CPU encode that holds the
# matching lock. Retrieval never looks deeper than the newest ~2k unscored jobs
# (search_for_resume corpus_cap), so a few thousand is ample. Kept SMALL because
# the embedding model runs on a shared CPU: encoding 25k long job texts on a
# contended box took 20+ min (an effective hang); 4k of shortened text is ~1 min.
REBUILD_MAX_JOBS = 4000

# Supabase enforces a statement_timeout, so a single UPDATE over thousands of
# rows gets cancelled ("QueryCanceled … N bound parameter sets"). Commit the
# embedding_id backfill in small batches so every statement stays fast.
_EMBED_UPDATE_BATCH = 500

# Same reasoning for reads: a single SELECT ... WHERE id IN (thousands) is one
# giant statement Supabase cancels via statement_timeout (seen as a 4000-param
# QueryCanceled that stalled matching for 330s). Fetch in ID chunks instead —
# the union of the batches is identical to one big query.
_DB_ID_CHUNK = 500


def _bulk_set_embedding_ids(mappings: list[dict]) -> None:
    """Write [{id, embedding_id}, …] back to the Job table in committed chunks."""
    if not mappings:
        return
    total = len(mappings)
    for start in range(0, total, _EMBED_UPDATE_BATCH):
        batch = mappings[start:start + _EMBED_UPDATE_BATCH]
        try:
            with get_session() as session:
                session.bulk_update_mappings(Job, batch)
                session.commit()
        except Exception as e:
            # One slow/failed batch must not abort the whole backfill — the FAISS
            # index is already written, so a missed embedding_id just means those
            # rows get re-indexed next run rather than losing data.
            log.warning("embedding_id batch %d-%d failed: %s", start, start + len(batch), e)
    log.info("Backfilled embedding_id for %d jobs in %d batches",
             total, (total + _EMBED_UPDATE_BATCH - 1) // _EMBED_UPDATE_BATCH)


def _profile_summary_chunk(profile) -> str:
    """Build the retrieval-steering summary chunk from THIS user's profile.
    Only includes fields the user actually filled in — never another
    candidate's roles, stack, or location."""
    lines: List[str] = []
    roles = (getattr(profile, "target_roles", "") or "").strip() or \
            (getattr(profile, "current_title", "") or "").strip()
    if roles:
        lines.append(f"Role Target: {roles}.")
    loc_parts = [
        (getattr(profile, "location", "") or "").strip(),
        (getattr(profile, "preferred_country", "") or "").strip(),
    ]
    loc = ", ".join(p for p in loc_parts if p)
    if loc:
        arrangement = "open to remote" if getattr(profile, "remote_ok", True) else "onsite/hybrid"
        lines.append(f"Preferred Location: {loc} ({arrangement}).")
    tech = (getattr(profile, "key_skills", "") or "").strip()
    if tech:
        lines.append(f"Key Technologies: {tech}.")
    summary = (getattr(profile, "professional_summary", "") or "").strip()
    if summary:
        lines.append(summary[:500])
    return "\n".join(lines)


def _legacy_summary_chunk() -> str:
    """Single-user/local fallback: search profile from the bundled Q&A store."""
    data = qa_resolver.data
    identity = data.get("identity", {})
    bg = data.get("background", {})
    pref = data.get("preferences", {})

    roles = "AI/ML Engineer, NLP Engineer, MLOps/Platform Engineer, or Backend Python Developer"
    tech = bg.get("tech_stack", "Python, PyTorch, TensorFlow, Scikit-learn, XGBoost, NLP, Generative AI, LLMs, RAG, LangChain, Multi-Agent Systems, Spark, PySpark, MLflow, Kubeflow, Vertex AI, BigQuery, Kafka, Airflow, AWS, GCP, Docker, Kubernetes, FastAPI, FAISS, PostgreSQL, MongoDB")
    loc = identity.get("location", "Cincinnati, OH")
    arrangement = pref.get("work_arrangement", "Open to remote, hybrid, or onsite")

    return (
        f"Role Target: {roles}.\n"
        f"Preferred Location: {loc} ({arrangement}).\n"
        f"Key Technologies: {tech}."
    )


def _chunk_resume(resume_text: str, profile=None) -> List[str]:
    """Split markdown resume by headers and append a target skills profile chunk.

    With a ``profile`` the summary chunk is built from that user's own target
    roles / skills / location; the bundled Q&A-store persona is ONLY used for
    legacy single-user (local) runs where no profile exists."""
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

    summary_chunk = _profile_summary_chunk(profile) if profile is not None else _legacy_summary_chunk()
    if summary_chunk:
        chunks.append(summary_chunk)
    return chunks


# Module-level model cache — the SentenceTransformer + CrossEncoder are heavy
# (hundreds of MB, seconds to load) and stateless once loaded, so we load them
# ONCE per process and share across every Matcher instance. Previously each
# Matcher() reloaded both models — extractor.py builds one per job, so discovery
# was reloading them dozens of times.
_MODEL_CACHE: dict = {}


def _device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def _get_embed_model():
    """Embedding model — always needed for FAISS retrieval."""
    if "embed" not in _MODEL_CACHE:
        device = _device()
        log.info("Loading embedding model %s on device: %s …", MODEL_NAME, device)
        _MODEL_CACHE["embed"] = SentenceTransformer(MODEL_NAME, device=device)
    return _MODEL_CACHE["embed"]


def _get_cross_encoder():
    """Cross-encoder — loaded lazily ONLY when the local rerank path runs.
    Under RERANK_PROVIDER=jina this is never loaded (no wasted ~200MB)."""
    if "cross" not in _MODEL_CACHE:
        device = _device()
        log.info("Loading cross-encoder model mixedbread-ai/mxbai-rerank-xsmall-v1 on device: %s …", device)
        _MODEL_CACHE["cross"] = CrossEncoder(
            "mixedbread-ai/mxbai-rerank-xsmall-v1",
            device=device,
            max_length=settings.cross_encoder_max_length,  # bound sequence length — CPU cost scales with it
        )
    return _MODEL_CACHE["cross"]


class Matcher:
    def __init__(self, user_id: str | None = None):
        self.model = _get_embed_model()  # cross-encoder loads lazily on first local rerank
        # Per-user index file. A single shared jobs.faiss was rebuilt per user,
        # so tenants overwrote each other's vectors and every pass triggered an
        # expensive full re-encode (the ~700s matching-lane pass seen in prod).
        # A file per user holds only that tenant's pool: small, incremental, no
        # cross-tenant thrash. Encode-only callers (embedding filter, extractor)
        # never persist, so they keep the default path.
        base: Path = settings.faiss_index_path
        if user_id and user_id not in ("local",):
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(user_id))
            self.index_path: Path = base.with_name(f"{base.stem}_{safe}{base.suffix}")
        else:
            self.index_path = base
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
        """Document text we embed for each job. Title weighted heavily.

        Only the first ~800 chars of the JD are embedded: CPU encode time scales
        with text length, and the role summary (title + opening of the JD) car-
        ries the semantic signal — the LLM reranker sees the full text later. The
        old 4000-char slice made a from-scratch rebuild pathologically slow on the
        shared CPU. ``or ""`` guards a null description (would crash the build)."""
        return f"{job.title}\n{job.title}\n{job.company} | {job.location}\n\n{(job.description or '')[:800]}"

    @staticmethod
    def _job_text_ce(job: Job, max_chars: int) -> str:
        """Short job text for the cross-encoder — title + opening of the JD only.
        Keeps each pair short so CPU scoring stays fast (cost scales with length)."""
        head = (job.description or "")[: max(0, max_chars - len(job.title) - 4)]
        return f"{job.title}\n\n{head}"

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

        # Cheap pass first: ids + embedding_id only. Loading every full Job row
        # (multi-KB descriptions) on every rebuild was a large, recurring DB +
        # memory cost when almost all jobs are usually already indexed.
        with get_session() as session:
            q = select(Job.id, Job.embedding_id).where(Job.is_closed == False)  # noqa: E712
            if user_id:
                q = q.where(Job.user_id == user_id)
            # Index only the NEWEST REBUILD_MAX_JOBS per user. Retrieval only ever
            # looks at the newest ~2k unscored jobs (search_for_resume) and
            # re-shortlists scored jobs via a direct DB query — so indexing every
            # historical posting is pure cost. A bloated pool (a role-less user
            # once adopted the whole shared pool → 115k jobs) turned the
            # from-scratch rebuild into a ~9-min CPU encode that held the matching
            # lock and stalled ALL matching. Bounding it keeps rebuilds fast.
            q = q.order_by(Job.first_seen.desc()).limit(REBUILD_MAX_JOBS)
            id_rows = session.exec(q).all()

        if not id_rows:
            log.warning("No jobs in DB to index.")
            return 0

        new_ids = [jid for jid, _emb in id_rows if jid not in existing_ids]
        updated_ids = [jid for jid, emb in id_rows if jid in existing_ids and emb is None]

        # If there are updated jobs, we must do a full rebuild to clean up stale vectors
        force_rebuild = len(updated_ids) > 0

        if not new_ids and not force_rebuild:
            log.info("All %d jobs are already indexed. No update needed.", len(id_rows))
            return len(id_rows)

        def _load_full(ids: list[int] | None) -> list[Job]:
            with get_session() as session:
                if ids is not None:
                    # Fetch by id in bounded chunks — one IN (...) over thousands
                    # of ids is a single statement Supabase cancels on timeout.
                    out: list[Job] = []
                    for start in range(0, len(ids), _DB_ID_CHUNK):
                        batch = ids[start:start + _DB_ID_CHUNK]
                        q = select(Job).where(Job.is_closed == False)  # noqa: E712
                        if user_id:
                            q = q.where(Job.user_id == user_id)
                        q = q.where(Job.id.in_(batch))
                        out.extend(session.exec(q).all())
                    return out
                # Full (from-scratch) build — same newest-first cap as the
                # cheap pass so a bloated pool can't produce a 100k-vector encode.
                q = select(Job).where(Job.is_closed == False)  # noqa: E712
                if user_id:
                    q = q.where(Job.user_id == user_id)
                q = q.order_by(Job.first_seen.desc()).limit(REBUILD_MAX_JOBS)
                return session.exec(q).all()

        if force_rebuild or existing_index is None or self.job_ids is None:
            # Build from scratch
            all_jobs = _load_full(None)
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
            
            # Save embedding_ids back to DB in CHUNKED bulk UPDATEs. A single
            # bulk_update_mappings over thousands of rows is one giant statement
            # that Supabase cancels via statement_timeout ("QueryCanceled …
            # 6376 bound parameter sets"); committing ~500 rows at a time keeps
            # every statement fast and well under the timeout.
            _bulk_set_embedding_ids(
                [{"id": int(j.id), "embedding_id": idx} for idx, j in enumerate(all_jobs)]
            )

            log.info("FAISS index built from scratch: %d vectors", len(all_jobs))
        else:
            # Incremental append — only the new jobs' full rows are loaded.
            new_jobs = _load_full(new_ids)
            log.info("Indexing %d new jobs incrementally...", len(new_jobs))
            new_texts = [self._job_text(j) for j in new_jobs]
            new_embs = self.encode(new_texts)
            
            start_idx = len(self.job_ids)
            existing_index.add(new_embs)
            new_ids_arr = np.array([j.id for j in new_jobs], dtype="int64")
            self.job_ids = np.concatenate([self.job_ids, new_ids_arr])
            faiss.write_index(existing_index, str(self.index_path))
            np.save(self.id_map_path, self.job_ids)
            
            # Save embedding_ids back to DB in CHUNKED bulk UPDATEs (see above).
            _bulk_set_embedding_ids(
                [{"id": int(j.id), "embedding_id": start_idx + idx} for idx, j in enumerate(new_jobs)]
            )

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

    def search_for_resume(self, resume_text: str, k: int = 30, user_id: str | None = None,
                          profile=None, only_unscored: bool = False,
                          corpus_cap: int = 2000) -> List[Tuple[int, float]]:
        """Hybrid search with RRF (Max-Similarity chunked query) + local Cross-Encoder reranking.

        ``only_unscored=True`` restricts the candidate corpus to jobs that have
        never been LLM-scored (``rerank_score IS NULL``), newest first. This is
        the freshness fix: ranking against the WHOLE historical pool let old,
        already-scored jobs win every cross-encoder slot, so brand-new postings
        were never ranked and the feed went stale. Already-scored jobs don't
        need retrieval at all — re-shortlisting them is a direct DB query in
        the pipeline. ``corpus_cap`` bounds BM25/CE cost as the pool grows.

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
            if only_unscored:
                q = (q.where(Job.rerank_score == None)  # noqa: E711
                      .order_by(Job.first_seen.desc())
                      .limit(corpus_cap))
            jobs = session.exec(q).all()

        if not jobs:
            return []

        # Filter out onsite jobs located in a different country than the user's
        # preferred one (same geo logic as rule_filter, so the two stages agree).
        # Legacy (no profile) keeps the original US-targeting default.
        preferred = norm_country(
            (getattr(profile, "preferred_country", "") or "United States") if profile else "United States"
        )

        filtered_jobs = []
        for j in jobs:
            loc_low = (j.location or "").lower()
            title_low = j.title.lower()

            is_outside = False
            # Remote roles can advertise a foreign HQ but still hire remote
            # candidates — never location-filter them (matches rule_filter).
            if not j.remote:
                haystack = loc_low if loc_low else title_low
                detected = detect_country(haystack)
                if detected and detected != preferred:
                    is_outside = True

            if not is_outside:
                filtered_jobs.append(j)

        jobs = filtered_jobs

        if not jobs:
            return []

        # Split resume into chunks to prevent vector dilution / sequence truncation
        chunks = _chunk_resume(resume_text, profile=profile)
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
        # Search the FULL index, not just top-len(jobs): the index holds every
        # open job while the corpus may be a small unscored subset — a shallow
        # search could rank only old (excluded) jobs and miss the corpus.
        chunk_embs = self.encode(chunks)
        faiss_scores, faiss_idxs = self.index.search(chunk_embs, self.index.ntotal)

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

        # Cross-encoder is the expensive CPU stage — cap how many pairs it scores.
        # BM25+FAISS+RRF already rank well; the cross-encoder only refines the top
        # slice. Capping candidates here (not at k) is the main CPU-cost lever.
        ce_cap = min(len(rrf_scores), settings.cross_encoder_cap)
        rrf_ranking = sorted(rrf_scores, key=lambda x: x[1], reverse=True)
        if only_unscored:
            # FRESHNESS RESERVE: the unscored corpus is already newest-first
            # (first_seen desc — rrf_scores preserves that order), but narrowing
            # it to the top ce_cap by RELEVANCE alone drops brand-new postings
            # that aren't among the most resume-similar — so they never get
            # scored and never reach the shortlist, even though they're already
            # in the pool (visible in All Jobs). The downstream fresh-first LLM
            # budget can only pick from what the cross-encoder scored, so we
            # guarantee the freshest postings a slice HERE. Cost-neutral.
            from app.matching.fresh_budget import reserve_fresh_slice
            top_candidates = reserve_fresh_slice(
                rrf_scores, rrf_ranking, ce_cap, key=lambda pair: pair[0].id,
            )
        else:
            top_candidates = rrf_ranking[:ce_cap]
        log.info("Sending %d candidates to cross-encoder (from %d total)", len(top_candidates), len(jobs))

        # 4. Reranking — use SHORT profile+job text so each pair stays well under
        # the model's max_length. CPU cost grows fast with sequence length, so we
        # truncate aggressively (title + opening of the JD carries the key signal);
        # the LLM reranker sees the full text later.
        _n = settings.cross_encoder_text_chars
        prof_short = profile_chunk[:_n]
        ce_docs = [self._job_text_ce(j, _n) for j, _ in top_candidates]

        # Try a hosted rerank backend (Jina) first — ~300-800ms vs ~2min on the
        # Railway CPU. Falls back to the local cross-encoder on any failure.
        from app.matching.rerank_backend import rerank_scores
        backend_scores = rerank_scores(prof_short, ce_docs)
        if backend_scores is not None:
            scores_norm = np.asarray(backend_scores, dtype="float32")
        else:
            pairs = [(prof_short, d) for d in ce_docs]
            logits = _get_cross_encoder().predict(pairs, show_progress_bar=True, batch_size=64)
            # Sigmoid to normalize logits to a 0-1 probability score
            scores_norm = 1.0 / (1.0 + np.exp(-logits))

        final_scores = []
        for (job, _), score in zip(top_candidates, scores_norm):
            final_scores.append((job.id, float(score)))

        # Return top k sorted by Cross-Encoder score descending
        final_ranking = sorted(final_scores, key=lambda x: x[1], reverse=True)
        log.info("Cross-encoder top scores: %s", [(s[1]) for s in final_ranking[:5]])
        return final_ranking[:k]
