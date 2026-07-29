from pathlib import Path


def safe_filename(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", "."))
    return cleaned.strip() or "audio.wav"

def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()
