from pathlib import Path

def safe_filename(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", ".")).strip() or "audio.wav"

def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()
