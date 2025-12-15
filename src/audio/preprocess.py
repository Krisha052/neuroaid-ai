from pathlib import Path
from typing import Optional

def ensure_wav_16k_mono(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    MVP placeholder.

    In a full build, use ffmpeg/pydub to:
    - convert to WAV
    - resample to 16kHz
    - convert to mono
    - normalize volume
    """
    # For MVP: assume uploaded audio is already WAV 16k mono.
    return input_path if output_path is None else output_path

