import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class AppConfig:
    sample_prompts_path: Path = PROJECT_ROOT / "data" / "sample_prompts.txt"
    sample_audio_dir: Path = PROJECT_ROOT / "data" / "sample_audio"
    model_dir: Path = PROJECT_ROOT / "models"
    model_path: Path = PROJECT_ROOT / "models" / "risk_model.joblib"
    reports_dir: Path = PROJECT_ROOT / "reports"

    whisper_model_size: str = "base"   # tiny/base/small
    sample_rate: int = 16000

    # Upload limits enforced by the API (Phase 5/6: reliability + privacy)
    max_upload_mb: int = 15
    max_duration_sec: float = 120.0

    spacy_model: str = "en_core_web_sm"

CONFIG = AppConfig()

# Where the Streamlit demo client reaches the Flask API. Override for Docker/remote setups.
API_BASE_URL = os.environ.get("NEUROAID_API_BASE_URL", "http://localhost:8000")

