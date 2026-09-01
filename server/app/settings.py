from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRRC_", env_file=".env", extra="ignore")

    secret_key: str = "development-only-change-me-32-characters"
    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str = ""
    database_path: Path = Path("./data/crrc-sop.sqlite3")
    evidence_dir: Path = Path("./data/evidence")
    access_token_minutes: int = 720
    root_path: str = ""

    def validate_production(self) -> None:
        if len(self.secret_key) < 32 or self.secret_key.startswith("development-"):
            raise RuntimeError("CRRC_SECRET_KEY must contain at least 32 non-default characters")
        if len(self.bootstrap_admin_password) < 12:
            raise RuntimeError("CRRC_BOOTSTRAP_ADMIN_PASSWORD must contain at least 12 characters")
