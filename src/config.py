from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class AppConfig:
    sample_prompts_path: Path = PROJECT_ROOT / "data" / "sample_prompts.txt"
    sample_audio_dir: Path = PROJECT_ROOT / "data" / "sample_audio"
    model_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    whisper_model_size: str = "base"   # tiny/base/small
    sample_rate: int = 16000

CONFIG = AppConfig()

