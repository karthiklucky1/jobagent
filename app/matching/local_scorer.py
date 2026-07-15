"""Local distilled scorer — the $0-per-job replacement for LLM fit scoring.

The play (see docs/DISTILLATION.md): every authoritative LLM score already paid
for is a training example. A small cross-encoder fine-tuned on those
(résumé, job) → score pairs runs on the existing CPU at ~30-100ms/job, so
scoring cost stops depending on user count entirely.

This module is the INFERENCE + SHADOW half:
- ``LocalScorer`` lazily loads a fine-tuned model from ``local_scorer_path``.
  No model on disk → everything here silently no-ops (safe default until the
  first Colab training run — see scripts/train_local_scorer.py).
- ``shadow_score`` runs the local model NEXT TO a fresh LLM final and records
  the (llm, local) pair as a FunnelEvent — zero user-facing effect. The
  recorded pairs are the evidence for/against flipping scoring to local:
  aggregate them with scripts/shadow_report.py.

The pair text fed to the model is built by ``build_pair`` — training
(scripts/train_local_scorer.py) imports the SAME builder, so the model always
sees identical formatting at train and inference time.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

from app.config import settings

log = logging.getLogger(__name__)

# Cross-encoder inputs are capped at 512 tokens total, so the pair packs the
# most decision-relevant content first. Slices are deliberately modest — the
# tokenizer truncates the tail anyway.
_RESUME_SLICE = 2000
_DESC_SLICE = 2500


def build_pair(resume_text: str, job) -> Tuple[str, str]:
    """The (résumé, job) text pair — SHARED between training and inference."""
    job_text = (f"{job.title} at {job.company} | {job.location} | "
                f"remote={bool(job.remote)}\n{(job.description or '')[:_DESC_SLICE]}")
    return (resume_text or "")[:_RESUME_SLICE], job_text


class LocalScorer:
    """Lazy singleton around the fine-tuned cross-encoder. Thread-safe."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._load_failed = False
        self._load_lock = threading.Lock()

    @classmethod
    def get(cls) -> "LocalScorer":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def available(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        return Path(settings.local_scorer_path).is_dir()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        with self._load_lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False
            path = Path(settings.local_scorer_path)
            if not path.is_dir():
                self._load_failed = True
                return False
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(str(path), max_length=512)
                log.info("LocalScorer: loaded distilled model from %s", path)
                return True
            except Exception as e:
                self._load_failed = True
                log.warning("LocalScorer: failed to load model at %s (%s) — disabled", path, e)
                return False

    def score(self, resume_text: str, job) -> Optional[float]:
        """0-100 fit score from the distilled model, or None when unavailable."""
        if not self._ensure_loaded():
            return None
        try:
            pair = build_pair(resume_text, job)
            raw = float(self._model.predict([pair])[0])
            return max(0.0, min(100.0, raw * 100.0))
        except Exception as e:
            log.debug("LocalScorer: predict failed for job %s: %s", getattr(job, "id", "?"), e)
            return None


# ── Shadow mode: local model runs beside LLM finals, agreement is recorded ────
_shadow_stats = {"n": 0, "abs_err_sum": 0.0, "within10": 0}
_shadow_lock = threading.Lock()
_SHADOW_LOG_EVERY = 25


def shadow_score(jid: int, resume_text: str, job, llm_score: float) -> None:
    """Best-effort: never raises, never affects the real scoring path."""
    try:
        if not settings.local_scorer_shadow:
            return
        scorer = LocalScorer.get()
        if not scorer.available():
            return
        local = scorer.score(resume_text, job)
        if local is None:
            return
        _record_shadow(jid, float(llm_score), local)
    except Exception as e:
        log.debug("shadow score failed for %s: %s", jid, e)


def _record_shadow(jid: int, llm_score: float, local_score: float) -> None:
    from app.db.init_db import get_session
    from app.db.models import FunnelEvent
    err = abs(llm_score - local_score)
    with _shadow_lock:
        _shadow_stats["n"] += 1
        _shadow_stats["abs_err_sum"] += err
        _shadow_stats["within10"] += 1 if err <= 10 else 0
        n = _shadow_stats["n"]
        if n % _SHADOW_LOG_EVERY == 0:
            log.info("Shadow scorer agreement (n=%d): MAE=%.1f, within-10pts=%.0f%%",
                     n, _shadow_stats["abs_err_sum"] / n,
                     100.0 * _shadow_stats["within10"] / n)
    try:
        with get_session() as session:
            session.add(FunnelEvent(
                job_id=jid, stage="shadow_score", passed=err <= 10,
                reason=f"llm={llm_score:.0f} local={local_score:.0f}",
                metadata_json=json.dumps({"llm": round(llm_score, 1),
                                          "local": round(local_score, 1)}),
            ))
            session.commit()
    except Exception as e:
        log.debug("shadow event write failed for %s: %s", jid, e)
