import logging
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.audio.diagnostics import wav_info
from src.audio.recorder import ephemeral_audio_file
from src.config import CONFIG
from src.model.predict import predict_risk
from src.nlp.feature_extractor import extract_features
from src.speech_to_text.text_cleaning import clean_transcript, tokenize
from src.speech_to_text.whisper_wrapper import transcribe

from .errors import (
    InvalidInputError,
    PayloadTooLargeError,
    UnprocessableAudioError,
)

logger = logging.getLogger("neuroaid.api")

def load_prompts(path: Path) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
            continue
        if current:
            sections[current].append(line)
    return sections

def _validate_upload(data: bytes) -> None:
    if not data:
        raise InvalidInputError("Uploaded file is empty.")
    max_bytes = CONFIG.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"Upload exceeds the {CONFIG.max_upload_mb}MB limit."
        )

def run_screening(
    audio_bytes: bytes,
    prompt_text: str,
    transcription_mode: str = "local",
) -> Dict[str, Any]:
    """
    Core pipeline shared by the API's /screen endpoint: validate input,
    transcribe, extract features (lexical + phoneme + acoustic), and -- if a
    trained model is available -- attach a plain-language risk assessment.

    Audio is written to a temp file only for the duration of processing and
    is always deleted afterward (see ephemeral_audio_file), never persisted.
    """
    _validate_upload(audio_bytes)

    with ephemeral_audio_file(audio_bytes) as audio_path:
        try:
            info = wav_info(audio_path)
        except (wave.Error, EOFError) as exc:
            raise UnprocessableAudioError(
                "Could not read the uploaded file as a WAV audio stream."
            ) from exc

        duration = float(info["duration_sec"])
        if duration > CONFIG.max_duration_sec:
            raise PayloadTooLargeError(
                f"Audio exceeds the {CONFIG.max_duration_sec:.0f}s duration limit."
            )
        if duration <= 0:
            raise UnprocessableAudioError("Audio file contains no frames.")

        try:
            tr = transcribe(audio_path, mode=transcription_mode,
                             model_size=CONFIG.whisper_model_size)
        except Exception as exc:
            logger.exception("Whisper transcription failed")
            raise UnprocessableAudioError(
                "Transcription failed for this audio file."
            ) from exc

        cleaned = clean_transcript(tr.text)
        spoken_words = tokenize(cleaned)
        prompt_words = tokenize(clean_transcript(prompt_text))

        feat_res = extract_features(
            prompt_words, spoken_words, duration_sec=duration, audio_path=audio_path
        )

    response: Dict[str, Any] = {
        "audio_diagnostics": info,
        "transcript": tr.text,
        "features": feat_res.features,
        "notes": feat_res.notes,
    }

    if CONFIG.model_path.exists():
        try:
            response["risk_assessment"] = predict_risk(CONFIG.model_path, feat_res.features)
        except Exception:
            logger.exception("Risk model inference failed")
            response["risk_assessment"] = None
            response["notes"]["risk_model_error"] = "Risk model inference failed; see server logs."
    else:
        response["risk_assessment"] = None
        response["notes"]["risk_model_missing"] = (
            f"No trained model at {CONFIG.model_path}; run "
            "scripts/generate_training_data.py then src/model/train.py to enable risk scoring."
        )

    return response

def resolve_prompt_text(prompt_set: Optional[str], prompt_text: Optional[str]) -> str:
    if prompt_text:
        return prompt_text
    prompts = load_prompts(CONFIG.sample_prompts_path)
    if not prompt_set:
        raise InvalidInputError("Provide either 'prompt_set' or 'prompt_text'.")
    if prompt_set not in prompts:
        raise InvalidInputError(
            f"Unknown prompt_set '{prompt_set}'. Available: {list(prompts.keys())}"
        )
    return " ".join(prompts[prompt_set])
