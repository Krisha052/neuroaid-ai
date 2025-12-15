from pathlib import Path
import wave

def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        return {
            "channels": wf.getnchannels(),
            "sample_rate": wf.getframerate(),
            "frames": wf.getnframes(),
            "duration_sec": wf.getnframes() / float(wf.getframerate())
        }

