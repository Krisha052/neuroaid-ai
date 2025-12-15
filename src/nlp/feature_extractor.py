from dataclasses import dataclass
from typing import Dict, Any, List
from .phoneme_mapper import words_to_phoneme_sequence

@dataclass
class FeatureResult:
    features: Dict[str, float]
    notes: Dict[str, Any]

def estimate_wpm(num_words: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return (num_words / duration_sec) * 60.0

def simple_word_error_rate(prompt_words: List[str], spoken_words: List[str]) -> float:
    """
    MVP proxy: compare overlap ratio (not true alignment).
    """
    if not prompt_words:
        return 0.0
    prompt_set = set(prompt_words)
    spoken_set = set(spoken_words)
    matched = len(prompt_set.intersection(spoken_set))
    return 1.0 - (matched / max(1, len(prompt_set)))

def phoneme_mismatch_rate(prompt_words: List[str], spoken_words: List[str]) -> float:
    """
    MVP proxy: compare phoneme sequence lengths (not true forced alignment).
    """
    p_seq = words_to_phoneme_sequence(prompt_words)
    s_seq = words_to_phoneme_sequence(spoken_words)
    if not p_seq:
        return 0.0
    return abs(len(p_seq) - len(s_seq)) / float(len(p_seq))

def extract_features(prompt_words: List[str], spoken_words: List[str], duration_sec: float) -> FeatureResult:
    wpm = estimate_wpm(len(spoken_words), duration_sec)
    wer = simple_word_error_rate(prompt_words, spoken_words)
    pmr = phoneme_mismatch_rate(prompt_words, spoken_words)

    features = {
        "wpm": float(wpm),
        "word_error_proxy": float(wer),
        "phoneme_mismatch_proxy": float(pmr),
        "spoken_word_count": float(len(spoken_words)),
        "prompt_word_count": float(len(prompt_words)),
    }

    notes = {
        "mvp_warning": "These are proxy features; upgrade to real alignment for stronger validity."
    }
    return FeatureResult(features=features, notes=notes)
