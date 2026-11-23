"""Runtime configuration, read from environment variables.

Every external service (database, object storage, MLflow) is optional and
addressed by URL, so the same code runs against the local docker compose
stack, a cloud provider, or nothing at all (file-only mode used by the
scheduled GitHub Actions job).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "tabularisai/multilingual-sentiment-analysis"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value if value not in ("", None) else default


@dataclass(frozen=True)
class Settings:
    nyt_api_key: str | None = field(default_factory=lambda: _env("NYT_API_KEY"))
    database_url: str | None = field(default_factory=lambda: _env("DATABASE_URL"))
    s3_endpoint_url: str | None = field(default_factory=lambda: _env("S3_ENDPOINT_URL"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "nyt-sentiment"))
    mlflow_tracking_uri: str | None = field(default_factory=lambda: _env("MLFLOW_TRACKING_URI"))
    mlflow_experiment: str = field(
        default_factory=lambda: _env("MLFLOW_EXPERIMENT", "nyt-sentiment-forecast")
    )
    sentiment_model: str = field(default_factory=lambda: _env("SENTIMENT_MODEL", DEFAULT_MODEL))
    forecast_horizon_days: int = field(
        default_factory=lambda: int(_env("FORECAST_HORIZON_DAYS", "30"))
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))


def get_settings() -> Settings:
    return Settings()
