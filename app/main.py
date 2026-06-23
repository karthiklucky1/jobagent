"""Single-process orchestrator: FastAPI + Telegram bot (local dev only).

Production (Railway) runs: uvicorn app.api.server:app
The FastAPI server has its own built-in asyncio scheduler that handles
discovery + matching every 6h, so we don't duplicate those jobs here.

This file is only used locally (python -m app.main) to also start the
Telegram bot and the registry harvester/validator on a schedule.
"""
from __future__ import annotations

import asyncio
import logging
import threading

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from app.api.server import app as fastapi_app
from app.config import settings
from app.telegram_bot.bot import build_app as build_tg
from app.analytics.reporter import FunnelReporter

log = logging.getLogger(__name__)


def run_harvester_sync():
    import asyncio
    from app.discovery.registry import run_validation_loop
    try:
        asyncio.run(run_validation_loop(limit=500))
    except Exception as e:
        log.error("Registry weekly harvest job failed: %s", e)


def run_validator_sync():
    import asyncio
    from app.discovery.registry import run_validation_loop
    try:
        asyncio.run(run_validation_loop())
    except Exception as e:
        log.error("Registry daily validator job failed: %s", e)


def run_profile_harvest_sync():
    from app.intelligence.harvester import run_harvest_all_users
    try:
        run_harvest_all_users()
    except Exception as e:
        log.error("Weekly profile harvest job failed: %s", e)


def start_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(daemon=True)
    # NOTE: discovery/matching/tailoring are intentionally NOT here.
    # server.py's asyncio _scheduler() handles those every 6h in both
    # local and production — adding them here would double the runs.
    # Harvester weekly (Sundays at 2 AM)
    sched.add_job(run_harvester_sync, "cron", day_of_week="sun", hour=2, minute=0, id="harvester")
    # Validator daily (Daily at 3 AM)
    sched.add_job(run_validator_sync, "cron", hour=3, minute=0, id="validator")
    # Weekly personal profile harvester (Mondays at 6 AM)
    sched.add_job(run_profile_harvest_sync, "cron", day_of_week="mon", hour=6, minute=0, id="profile_harvester")
    # Daily funnel report at 8 PM local
    sched.add_job(FunnelReporter.send_daily_report, "cron", hour=20, minute=0, id="funnel_report")
    sched.start()
    log.info("Scheduler started (local dev — discovery handled by server.py).")
    return sched


def start_api() -> None:
    uvicorn.run(fastapi_app, host=settings.api_host, port=settings.api_port, log_level="info")


def start_bot() -> None:
    """python-telegram-bot needs its own event loop on a non-main thread."""
    asyncio.set_event_loop(asyncio.new_event_loop())
    tg = build_tg()
    tg.run_polling(close_loop=False, stop_signals=())


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    start_scheduler()
    threading.Thread(target=start_bot, daemon=True, name="tg-bot").start()
    start_api()


if __name__ == "__main__":
    main()
