# Distilled Local Scorer — runbook

Goal: stop paying per-job LLM prices for fit scoring. Every LLM score already
in the DB is a training example; a small cross-encoder fine-tuned on them runs
on the existing CPU at ~30-100ms/job for $0. The LLM's remaining jobs: writing
the reasoning users see on shortlist cards, and auditing the student.

This is the documented industry pattern — Indeed fine-tuned a smaller GPT on
GPT-4 outputs (same quality, 60% fewer tokens, ~20M msgs/day); LinkedIn
distills cross-encoder teachers into cheap two-tower students for serving.

## Pipeline

```
1. EXPORT  (on the server / against prod DB)
   python -m scripts.export_training_data --out data/training/scoring_distill.jsonl
   → only genuine LLM finals are exported (ghost/prescore/rule/door-stamped
     rows are excluded — their labels would teach the model the cheap gates,
     not the rubric). Contains résumés = PII: keep out of git, delete after use.

2. TRAIN   (free Colab GPU, ~1-2h)
   Upload scripts/train_local_scorer.py + the JSONL to Colab:
     !pip install -q sentence-transformers
     !python train_local_scorer.py --data scoring_distill.jsonl --out hirepath-scorer
   Prints validation MAE / within-10 vs the LLM teacher.
   Optional: pad thin domains with free HF data (cnamuangtoun/
   resume-job-description-fit, netsol/resume-score-details) mapped to the same
   JSONL fields before training.

3. DEPLOY THE SHADOW  (no user-facing change)
   Place the model directory at data/models/hirepath-scorer (LOCAL_SCORER_PATH).
   LOCAL_SCORER_SHADOW is on by default: every scoring-lane LLM final also runs
   the local model and records agreement (FunnelEvent stage="shadow_score";
   "Shadow scorer agreement" lines appear in logs every 25 finals).

4. DECIDE  (after ~a week of shadow data)
   python -m scripts.shadow_report --days 7
   Flip signal: shortlist-decision agreement ≥ 90% sustained. Until then the
   model costs nothing and changes nothing.

5. FLIP    (separate change, deliberately not built yet)
   Local model becomes Tier-2 for ranking; LLM writes reasoning only for
   shortlisted top-N and audits ~50 random jobs/day (those audits are the next
   retraining batch). Build this only after step 4's numbers justify it.
```

## Consistency invariant

`build_pair()` in app/matching/local_scorer.py and its copy in
scripts/train_local_scorer.py MUST stay byte-identical — the model must see the
same text formatting at train and inference time. If you change one, change
both and retrain.

## Retraining cadence

Monthly, or when shadow_report shows drift (agreement falling): re-export
(the audit finals since last training are new lessons), retrain, redeploy the
model dir. Each cycle makes the student better on domains it was weak in.
