"""
Application settings loaded from environment variables via pydantic-settings.
Copy .env.example to .env and fill in the values.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application
    APP_NAME: str = "QYX MedTech CASA"
    APP_VERSION: str = "0.2.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # Security
    SECRET_KEY: str = "changeme-replace-with-secure-random-secret"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # CORS (only needed when the Vite dev server runs separately)
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # CASA pipeline
    TRAINING_DIR: Path = REPO_DIR / "training"            # the pipeline's source (src/) lives here
    PIPELINE_CONFIG: Path = REPO_DIR / "training" / "configs" / "default.yaml"
    VALIDATION_DIR: Path = REPO_DIR / "training" / "outputs" / "seg" / "validation"
    DATA_DIR: Path = BACKEND_DIR / "data"                 # uploaded videos + per-analysis outputs
    MAX_UPLOAD_MB: int = 500
    ALLOWED_VIDEO_EXTENSIONS: list[str] = [".mp4", ".avi", ".mov", ".mkv"]

    # Built frontend (served by FastAPI when present: `npm run build` in frontend/)
    FRONTEND_DIST: Path = REPO_DIR / "frontend" / "dist"

    # Logging
    LOG_LEVEL: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
