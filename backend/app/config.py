"""Application configuration.

All settings come from the environment (prefix ``QIE_``) or an optional ``.env`` file —
nothing host-specific is hard-coded, keeping the cloud/multi-user door open (DESIGN §4.1a).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from env / ``.env``.

    Every binary path here is a *root* for the storage provider or model cache; the DB
    stores storage **keys**, never absolute paths (DESIGN §4.1a / §6).
    """

    model_config = SettingsConfigDict(
        env_prefix="QIE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Identity / server ---
    app_name: str = "QIE Studio"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Persistence ---
    # psycopg v3 sync driver. Compose wires this to the `db` service.
    database_url: str = "postgresql+psycopg://qie:qie@localhost:5432/qie"
    run_migrations_on_startup: bool = True

    # --- Storage provider (DESIGN §4.1a) ---
    storage_backend: str = "local"  # "local" now; "s3" later
    data_root: Path = Path("/data")  # LocalFsStorage root: assets/ outputs/ loras/ thumbs/

    # --- Model cache ---
    models_cache: Path = Path("/models")  # HF hub cache (HF_HOME)

    # --- Device / precision (DESIGN §9) ---
    device_override: str | None = None  # force "cuda"/"rocm"/"mps"/"cpu"; else auto-detect
    default_precision: str = "bf16"  # bf16 | fp8 | int4
    warmup_on_startup: bool = False

    # --- Default model ids (config-driven, overridable at run time; DESIGN §2) ---
    default_edit_model: str = "Qwen/Qwen-Image-Edit-2511"
    default_generate_model: str = "Qwen/Qwen-Image"
    default_rewriter_model: str = "Qwen/Qwen3-VL-8B-Instruct"

    # --- Live latent previews (DESIGN §5.5) ---
    preview_every_n_steps: int = 5  # throttled latent->RGB decode cadence; 0 disables

    # --- Secrets ---
    # Used to derive the Fernet key that encrypts integration API keys at rest (DESIGN §5.7).
    # MUST be overridden in any real deployment; a dev default is provided so the app boots.
    secret_key: str = "dev-insecure-change-me-please-32bytes!!"

    # --- Frontend ---
    # Directory of built SPA static assets served by the backend. Optional in dev.
    frontend_dist: Path = Path("frontend_dist")

    # --- Defaults for outputs ---
    default_output_format: str = "png"  # png | webp | jpeg
    default_output_quality: int = 95

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Allow a comma-separated string for CORS origins from a single env var."""
        if isinstance(v, str) and not v.startswith("["):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def is_local_storage(self) -> bool:
        return self.storage_backend == "local"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (one per process)."""
    return Settings()
