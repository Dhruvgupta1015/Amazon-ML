"""config.py - Application Configuration."""
from __future__ import annotations
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # Data paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data_raw/student_resource/dataset"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "output"))
    MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "models"))

    # Pipeline
    USE_DENSE_BLOCKING: bool = True
    MAX_CANDIDATES_PER_S1: int = 50
    LSH_THRESHOLD: float = 0.25
    LSH_PERMUTATIONS: int = 128
    ANN_TOP_K: int = 30
    DENSE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    NEG_PER_POS: int = 5
    DEFAULT_THRESHOLD: float = 0.5

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
