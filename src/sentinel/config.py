import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    data_dir: Path
    llm_timeout: float = 180

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "sentinel.db"


def get_settings() -> Settings:
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1").rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", "local"),
        llm_model=os.getenv("LLM_MODEL", "qwen3.6-35b-a3b"),
        data_dir=Path(os.getenv("SENTINEL_DATA", str(ROOT / "data"))),
    )
