from pathlib import Path
from typing import Optional

def save_uploaded_audio(uploaded_file, out_dir: Path, filename: str = "sample.wav") -> Optional[Path]:
    """
    Save an uploaded Streamlit file to disk and return its path.
    Expects WAV for MVP reliability.
    """
    if uploaded_file is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path

