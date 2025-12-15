from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class TranscriptResult:
    text: str
    segments: Optional[list] = None
    meta: Optional[Dict[str, Any]] = None

def transcribe_local_whisper(audio_path: Path, model_size: str = "base") -> TranscriptResult:
    """
    Local Whisper transcription.
    Requires `openai-whisper` and a working ffmpeg install for most formats.
    """
    import whisper  # local import so app can run without whisper installed

    model = whisper.load_model(model_size)
    out = model.transcribe(str(audio_path))
    return TranscriptResult(
        text=out.get("text", "").strip(),
        segments=out.get("segments"),
        meta={"language": out.get("language")}
    )

def transcribe(audio_path: Path, mode: str = "local", model_size: str = "base") -> TranscriptResult:
    """
    mode:
      - "local": local whisper
      - "none": skip transcription (for text-only testing)
    """
    if mode == "none":
        return TranscriptResult(text="")
    if mode == "local":
        return transcribe_local_whisper(audio_path, model_size=model_size)
    raise ValueError(f"Unsupported transcription mode: {mode}")
