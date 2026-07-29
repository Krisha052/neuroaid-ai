import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def ephemeral_audio_file(data: bytes, suffix: str = ".wav") -> Iterator[Path]:
    """
    Write uploaded audio bytes to a private temp file for the duration of
    processing, then delete it -- regardless of whether processing succeeds.

    This replaces the old behavior of persisting every upload to
    data/sample_audio/uploaded.wav, which contradicted the "no permanent
    storage" claim in docs/ETHICS_PRIVACY.md.
    """
    fd, path_str = tempfile.mkstemp(suffix=suffix)
    path = Path(path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        yield path
    finally:
        path.unlink(missing_ok=True)
