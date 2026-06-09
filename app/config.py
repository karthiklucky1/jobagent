"""Centralized config loaded from .env."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    tavily_api_key: str = ""
    exa_api_key: str = ""

    # Job board APIs
    serpapi_key: str = ""            # serpapi.com — Google Jobs (LinkedIn/Indeed/Glassdoor). Free: 100/mo
    remotive_enabled: bool = True    # Remotive public API — no key needed
    remoteok_enabled: bool = True    # RemoteOK public API — no key needed
    max_jobs_per_source: int = 50    # Cap per source per discovery run

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Local personal dashboard
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Paths
    data_dir: Path = Path("./data")
    resume_path: Path = Path("./data/resume_master.md")
    resume_docx_path: Path = Path("./data/resume_master.docx")
    profiles_dir: Path = Path("./data/profiles")
    faiss_index_path: Path = Path("./data/jobs.faiss")
    sqlite_path: Path = Path("./data/jobagent.db")
    bootstrap_path: Path = Path("./data/bootstrap_companies.json")

    # Applicant
    applicant_first_name: str = "Karthik"
    applicant_last_name: str = ""
    applicant_email: str = ""
    applicant_phone: str = ""
    applicant_location: str = "Cincinnati, OH"
    applicant_github: str = ""
    applicant_linkedin: str = ""
    applicant_work_auth: str = ""

    # Matching
    min_match_score: float = 0.20
    top_k_rerank: int = 500
    daily_apply_limit: int = 25

    # Models
    scoring_model: str = "claude-3-5-haiku-20241022"
    tailoring_model: str = "claude-sonnet-4-6"
    doctor_model: str = "claude-haiku-4-5-20251001"   # cheap Haiku for Doctor LLM verdict

    # Thresholds & Constraints
    min_embedding_score: float = 0.35
    qa_confidence_threshold: float = 0.7
    grounding_similarity_threshold: float = 0.5

    ghost_score_threshold: float = 0.6   # jobs at or above this score are skipped as likely ghost postings

    # Submission Delays & Limits
    submission_jitter_min: float = 180.0
    submission_jitter_max: float = 480.0
    headless: bool = True

    # Discovery
    greenhouse_boards: str = ""
    lever_boards: str = ""
    ashby_boards: str = ""
    jobs_keywords: str = "Machine Learning Engineer,AI Engineer,Python Developer,LLM Engineer,AI/ML Engineer,Backend Python Engineer"

    @property
    def jobs_keywords_list(self) -> List[str]:
        return [k.strip() for k in self.jobs_keywords.split(",") if k.strip()]

    @property
    def greenhouse_boards_list(self) -> List[str]:
        return [b.strip() for b in self.greenhouse_boards.split(",") if b.strip()]

    @property
    def lever_boards_list(self) -> List[str]:
        return [b.strip() for b in self.lever_boards.split(",") if b.strip()]

    @property
    def ashby_boards_list(self) -> List[str]:
        return [b.strip() for b in self.ashby_boards.split(",") if b.strip()]

    @property
    def sqlite_url(self) -> str:
        return f"sqlite:///{self.sqlite_path}"


settings = Settings()
