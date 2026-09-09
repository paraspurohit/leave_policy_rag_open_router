"""Application configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Configuration used by the RAG pipeline."""

    openrouter_api_key: str
    openrouter_base_url: str
    embedding_model: str
    generation_model: str
    raw_data_dir: Path
    processed_data_dir: Path


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


settings = Settings(
    openrouter_api_key=_required_env("OPENROUTER_API_KEY"),
    openrouter_base_url=os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    ),
    embedding_model=_required_env("EMBEDDING_MODEL"),
    generation_model=_required_env("GENERATION_MODEL"),
    raw_data_dir=PROJECT_ROOT / "data" / "raw",
    processed_data_dir=PROJECT_ROOT / "data" / "processed",
)
