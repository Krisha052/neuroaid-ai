from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.audio.diagnostics import wav_info
from src.config import CONFIG
from src.model.predict import predict_risk as run_predict_risk
from src.nlp.feature_extractor import extract_features as run_extract_features
from src.speech_to_text.text_cleaning import clean_transcript, tokenize
from src.speech_to_text.whisper_wrapper import transcribe as run_transcribe


@dataclass
class ScreeningContext:
    """
    Mutable state the tools read/write as the agent calls them. Intermediate
    results (transcript, features, ...) live here rather than being passed
    back and forth through the LLM's tool-call arguments -- keeps tool
    schemas small and avoids relying on the model to faithfully echo large
    values.
    """
    audio_path: Path
    prompt_text: str
    transcription_mode: str = "local"
    duration_sec: float = 0.0
    transcript: Optional[str] = None
    prompt_words: List[str] = field(default_factory=list)
    spoken_words: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Optional[Dict[str, Any]] = None

def tool_transcribe(ctx: ScreeningContext) -> Dict[str, Any]:
    info = wav_info(ctx.audio_path)
    ctx.duration_sec = float(info["duration_sec"])

    tr = run_transcribe(ctx.audio_path, mode=ctx.transcription_mode,
                         model_size=CONFIG.whisper_model_size)
    ctx.transcript = tr.text
    ctx.prompt_words = tokenize(clean_transcript(ctx.prompt_text))
    ctx.spoken_words = tokenize(clean_transcript(tr.text))

    return {
        "transcript": tr.text,
        "duration_sec": ctx.duration_sec,
        "spoken_word_count": len(ctx.spoken_words),
        "prompt_word_count": len(ctx.prompt_words),
    }

def tool_extract_features(ctx: ScreeningContext) -> Dict[str, Any]:
    if ctx.transcript is None:
        raise ValueError("transcribe must be called before extract_features")

    result = run_extract_features(
        ctx.prompt_words, ctx.spoken_words,
        duration_sec=ctx.duration_sec, audio_path=ctx.audio_path,
    )
    ctx.features = result.features
    ctx.notes.update(result.notes)
    return result.features

def tool_predict_risk(ctx: ScreeningContext) -> Dict[str, Any]:
    if not ctx.features:
        raise ValueError("extract_features must be called before predict_risk")
    if not CONFIG.model_path.exists():
        return {"available": False, "message": "No trained risk model is available."}

    ctx.risk_assessment = run_predict_risk(CONFIG.model_path, ctx.features)
    return {"available": True, **ctx.risk_assessment}

TOOL_EXECUTORS = {
    "transcribe": tool_transcribe,
    "extract_features": tool_extract_features,
    "predict_risk": tool_predict_risk,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "transcribe",
        "description": "Transcribe the uploaded audio with Whisper. Always call this first.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "extract_features",
        "description": (
            "Extract lexical (word error rate), phoneme-mismatch, and acoustic "
            "(MFCC / pause-ratio) features by comparing the transcript to the "
            "reading prompt. Call this after transcribe."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "predict_risk",
        "description": (
            "Run the trained risk model over the extracted features to get a "
            "risk score and band. Call this after extract_features, only if the "
            "sample looks analyzable."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "recommend_retake",
        "description": (
            "Call this INSTEAD of predict_risk/finalize_assessment if the sample "
            "isn't analyzable -- e.g. an empty or near-empty transcript, a "
            "degenerate duration, or a tool error indicating the audio couldn't "
            "be processed. Recommends the user retake the reading rather than "
            "forcing a risk assessment on bad input."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
    {
        "name": "finalize_assessment",
        "description": (
            "Submit the final, plain-language explanation for the user once "
            "transcribe, extract_features, and predict_risk have all been "
            "called. This ends the screening."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": (
                        "A tailored, human-readable explanation of the result "
                        "for a non-technical parent or teacher."
                    ),
                },
            },
            "required": ["explanation"],
        },
    },
]
