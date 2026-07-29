from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .phoneme_mapper import words_to_phoneme_sequence
from .spacy_pipeline import word_error_rate


@dataclass
class FeatureResult:
    features: Dict[str, float]
    notes: Dict[str, Any]

def estimate_wpm(num_words: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return (num_words / duration_sec) * 60.0

def phoneme_mismatch_rate(prompt_words: List[str], spoken_words: List[str]) -> float:
    """
    MVP proxy: compare phoneme sequence lengths (not true forced alignment).
    Real forced phoneme alignment (e.g. via a CTC/Viterbi aligner) would be a
    stronger signal than this length-difference proxy -- tracked as a known
    limitation rather than overstated as full alignment.
    """
    p_seq = words_to_phoneme_sequence(prompt_words)
    s_seq = words_to_phoneme_sequence(spoken_words)
    if not p_seq:
        return 0.0
    return abs(len(p_seq) - len(s_seq)) / float(len(p_seq))

def extract_features(
    prompt_words: List[str],
    spoken_words: List[str],
    duration_sec: float,
    audio_path: Optional[Path] = None,
) -> FeatureResult:
    wpm = estimate_wpm(len(spoken_words), duration_sec)
    wer = word_error_rate(prompt_words, spoken_words)
    pmr = phoneme_mismatch_rate(prompt_words, spoken_words)

    features = {
        "wpm": float(wpm),
        "word_error_rate": float(wer),
        "phoneme_mismatch_proxy": float(pmr),
        "spoken_word_count": float(len(spoken_words)),
        "prompt_word_count": float(len(prompt_words)),
    }

    notes = {
        "phoneme_mismatch_proxy_warning": (
            "phoneme_mismatch_proxy compares phoneme-sequence length, not true "
            "forced alignment; treat as a coarse signal."
        )
    }

    if audio_path is not None:
        from src.audio.acoustic_features import extract_acoustic_features
        try:
            features.update(extract_acoustic_features(audio_path))
        except Exception as exc:
            notes["acoustic_features_error"] = str(exc)

    return FeatureResult(features=features, notes=notes)
