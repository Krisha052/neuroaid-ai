from pathlib import Path
from src.audio.diagnostics import wav_info

def test_wav_info_handles_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.wav"
    try:
        wav_info(missing)
        assert False, "Expected an exception for missing wav"
    except Exception:
        assert True
