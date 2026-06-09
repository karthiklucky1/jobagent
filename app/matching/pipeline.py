"""End-to-end matching pipeline.

1. Load resume from disk.
2. Rebuild FAISS index over all jobs (cheap, runs in seconds for <10k jobs).
3. Search top-K by cosine similarity.
4. Rerank with Claude, store score + reasoning back on Job rows.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

from sqlmodel import select

from app.config import settings
from app.db.init_db import get_session
from app.db.models import Application, ApplicationStatus, Job, JobSource
from app.matching.matcher import Matcher
from app.matching.reranker import Reranker
from app.matching.filters import RuleFilter, EmbeddingFilter, score_ghost
from app.intelligence.senior_reviewer import SeniorReviewer

# Sources where the bot can fill the form automatically
_AUTOFILL_SOURCES = {JobSource.GREENHOUSE, JobSource.LEVER, JobSource.ASHBY, JobSource.WORKDAY, JobSource.SMARTRECRUITERS}

log = logging.getLogger(__name__)


def _load_resume() -> str:
    p: Path = settings.resume_path
    if not p.exists():
        raise FileNotFoundError(
            f"Resume not found at {p}. Put a markdown version of your resume there."
        )
    return p.read_text(encoding="utf-8")


def _run_senior_review(reviewer: SeniorReviewer, job_id: int, app_id: int) -> None:
    """Call SeniorReviewer for one job and write results back to the Application row."""
    try:
        with get_session() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            result = reviewer.review(job)
            if result is None:
                return
            app = session.get(Application, app_id)
            if not app:
                return
            app.profile_variant = result.recommended_resume_variant
            app.senior_fit_score = float(result.fit_score)
            app.senior_verdict = result.senior_reviewer_verdict
            app.custom_highlight_block = result.custom_highlight_block or None
            session.add(app)
            session.commit()
            log.info(
                "SeniorReview job %d app %d: variant=%s score=%d genuine=%s",
                job_id, app_id, result.recommended_resume_variant,
                result.fit_score, result.is_genuine_match,
            )
    except Exception as e:
        log.exception("SeniorReview failed for job %d app %d: %s", job_id, app_id, e)


def run_matching() -> List[int]:
    resume = _load_resume()
    matcher = Matcher()
    matcher.rebuild()

    candidates = matcher.search_for_resume(resume, k=settings.top_k_rerank)
    candidates = [(jid, score) for jid, score in candidates if score >= settings.min_match_score]
    log.info("%d candidates above cross-encoder threshold %.2f", len(candidates), settings.min_match_score)

    rule_filter = RuleFilter()
    embedding_filter = EmbeddingFilter(matcher=matcher)
    reranker = Reranker()
    senior_reviewer = SeniorReviewer()
    shortlisted: List[int] = []

    # Count applications already created today to honour the daily cap in a short-lived session
    with get_session() as session:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_count = len(session.exec(
            select(Application).where(Application.created_at >= today_start)
        ).all())

    for jid, sim in candidates:
        with get_session() as session:
            job = session.get(Job, jid)
            if not job:
                continue

            # Check if job is already scored to avoid wasting LLM tokens and time
            if job.rerank_score is not None:
                if job.rerank_score >= 60:
                    existing = session.exec(
                        select(Application).where(Application.job_id == job.id)
                    ).first()
                    if not existing:
                        if today_count < settings.daily_apply_limit:
                            _track = "autofill" if job.source in _AUTOFILL_SOURCES else "manual"
                            session.add(
                                Application(
                                    job_id=job.id,
                                    status=ApplicationStatus.SHORTLISTED,
                                    apply_url=job.url,
                                    apply_track=_track,
                                )
                            )
                            shortlisted.append(job.id)
                            today_count += 1
                            session.commit()
                            log.info("Job '%s' @ '%s' already scored (%d) — %s track. Shortlisted.", job.title, job.company, job.rerank_score, _track)
                        else:
                            log.info("Daily apply limit reached — skipping application creation for already scored job %s.", job.title)
                    else:
                        log.info("Job '%s' @ '%s' already scored (%d) and has application. Skipping.", job.title, job.company, job.rerank_score)
                else:
                    log.info("Job '%s' @ '%s' already scored (%d). Skipping.", job.title, job.company, job.rerank_score)
                continue

            # 1. Rule Filter
            rule_res = rule_filter.filter(job)
            if not rule_res.passed:
                log.info("Job '%s' @ '%s' filtered by rules: %s", job.title, job.company, rule_res.reason)
                job.similarity_score = sim
                job.rerank_score = float(rule_res.score_override or 10.0)
                job.rerank_reasoning = f"Rule filtered: {rule_res.reason}"
                session.add(job)
                session.commit()
                continue

            # 2. Ghost Job Detection — cheap DB+text check, runs before LLM/embedding to save cost
            ghost_res = score_ghost(job, session)
            job.ghost_score = ghost_res.ghost_score
            job.ghost_flags = ghost_res.flags_json
            if ghost_res.is_ghost:
                log.info(
                    "Job '%s' @ '%s' flagged as likely ghost (score=%.2f flags=%s) — skipping",
                    job.title, job.company, ghost_res.ghost_score, ghost_res.flags,
                )
                job.similarity_score = sim
                job.rerank_score = 5.0
                job.rerank_reasoning = f"Ghost filtered (score={ghost_res.ghost_score:.2f}): {', '.join(ghost_res.flags)}"
                session.add(job)
                session.commit()
                continue
            elif ghost_res.ghost_score >= 0.3:
                log.warning(
                    "Job '%s' @ '%s' is suspicious (ghost_score=%.2f flags=%s) — proceeding with caution",
                    job.title, job.company, ghost_res.ghost_score, ghost_res.flags,
                )
            session.add(job)  # persist ghost_score even for passing jobs

            # 3. Embedding Filter
            emb_passed, emb_score, emb_reason = embedding_filter.filter(job, resume)
            if not emb_passed:
                log.info("Job '%s' @ '%s' filtered by embedding similarity: %s", job.title, job.company, emb_reason)
                job.similarity_score = sim
                job.rerank_score = 15.0
                job.rerank_reasoning = f"Embedding filtered: {emb_reason}"
                session.add(job)
                session.commit()
                continue

            # 3. LLM Reranking
            score, reason, concerns = reranker.score(resume, job)
            job.similarity_score = sim
            job.rerank_score = score
            job.rerank_reasoning = reason + (
                ("\nConcerns: " + "; ".join(concerns)) if concerns else ""
            )
            session.add(job)

            # If rerank ≥60, create an Application row in SHORTLISTED state
            new_app_id: int | None = None
            if score >= 60:
                existing = session.exec(
                    select(Application).where(Application.job_id == job.id)
                ).first()
                if not existing:
                    if today_count < settings.daily_apply_limit:
                        # ── Company cap: max 2 active applications per company ──
                        active_for_company = session.exec(
                            select(Application).join(Job, Application.job_id == Job.id)
                            .where(Job.company == job.company)
                            .where(Application.status.in_([
                                ApplicationStatus.SHORTLISTED,
                                ApplicationStatus.TAILORED,
                                ApplicationStatus.AUTOFILLED,
                                ApplicationStatus.AWAITING_USER,
                                ApplicationStatus.READY_TO_SUBMIT,
                            ]))
                        ).all()
                        if len(active_for_company) >= 2:
                            # Only bump in if this score beats the weakest existing slot
                            weakest = min(active_for_company,
                                         key=lambda a: session.get(Job, a.job_id).rerank_score or 0)
                            weakest_score = session.get(Job, weakest.job_id).rerank_score or 0
                            if score > weakest_score:
                                weakest.status = ApplicationStatus.SKIPPED
                                weakest.notes = f"Displaced by higher-scoring role (score={score:.0f} vs {weakest_score:.0f}) — company cap enforced"
                                session.add(weakest)
                                log.info("Company cap: displaced app %d (score=%.0f) for %s @ %s (score=%.0f)",
                                         weakest.id, weakest_score, job.title, job.company, score)
                            else:
                                log.info("Company cap: already have 2 active apps at %s, new score %.0f doesn't beat weakest %.0f — skipping",
                                         job.company, score, weakest_score)
                                session.commit()
                                continue

                        _track = "autofill" if job.source in _AUTOFILL_SOURCES else "manual"
                        new_app = Application(
                            job_id=job.id,
                            status=ApplicationStatus.SHORTLISTED,
                            apply_url=job.url,
                            apply_track=_track,
                        )
                        session.add(new_app)
                        session.flush()  # populate new_app.id before commit
                        new_app_id = new_app.id
                        shortlisted.append(job.id)
                        today_count += 1
                    else:
                        log.info("Daily apply limit (%d) reached — skipping application creation for job %s.", settings.daily_apply_limit, job.title)

            session.commit()
            log.info("Job %s @ %s: sim=%.3f rerank=%.0f — %s",
                     job.title, job.company, sim, score, reason)

        # Run SeniorReviewer outside the session (LLM call — don't hold a lock)
        if new_app_id is not None:
            _run_senior_review(senior_reviewer, jid, new_app_id)
    return shortlisted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    new_ids = run_matching()
    print(f"Shortlisted {len(new_ids)} new applications.")
