"""
Generate a training CSV for src/model/train.py.

No dyslexia-labeled speech dataset is freely available for a project like this
(real ones -- e.g. CMU Kids Corpus, OGI Kids' Speech -- sit behind paid LDC
licenses or IRB-gated research agreements, appropriately, since it's pediatric
health data). So this script takes a hybrid approach, documented end-to-end:

  - REAL acoustic features (MFCCs, pause ratio) are extracted from real
    open-licensed read-speech audio (data/sample_audio/real_samples, LibriSpeech
    dev-clean, CC BY 4.0 -- see that folder's README for attribution).
  - REAL lexical/phoneme features (WER, phoneme mismatch) are computed with the
    project's actual spaCy + CMU Pronouncing Dictionary pipeline, run against
    SYNTHETICALLY perturbed transcripts (random deletions/substitutions/filler
    insertions) to simulate a range of reading-fluency profiles, since 5 clips
    alone can't supply that variation.
  - The training LABEL is a documented, rule-based heuristic over those
    features, not a real clinical judgment. It exists to make the sklearn
    pipeline (train -> evaluate -> serve) run end-to-end, NOT to claim
    validated diagnostic performance. This is disclosed in the README, the
    docs, and in every API response that returns a risk score.

Usage: python -m scripts.generate_training_data
"""
import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.audio.acoustic_features import extract_acoustic_features
from src.audio.diagnostics import wav_info
from src.nlp.feature_extractor import estimate_wpm, phoneme_mismatch_rate
from src.nlp.spacy_pipeline import word_error_rate
from src.speech_to_text.text_cleaning import clean_transcript, tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_AUDIO_DIR = PROJECT_ROOT / "data" / "sample_audio" / "real_samples"
OUT_CSV = PROJECT_ROOT / "data" / "generated" / "training_data.csv"

VARIANTS_PER_CLIP = 40
SEED = 42
LABEL_NOISE_RATE = 0.08  # flip 8% of labels: avoids a trivially perfect, circular-looking fit

FILLER_WORDS = ["um", "uh", "like", "the", "and"]

def perturb_words(words: List[str], rng: random.Random) -> List[str]:
    """Simulate a spoken attempt at varying fluency: random deletions,
    substitutions, and filler insertions relative to the prompt."""
    p_del = rng.uniform(0.0, 0.4)
    p_sub = rng.uniform(0.0, 0.3)
    p_ins = rng.uniform(0.0, 0.2)

    out: List[str] = []
    for w in words:
        if rng.random() < p_del:
            continue
        out.append(rng.choice(FILLER_WORDS) if rng.random() < p_sub else w)
        if rng.random() < p_ins:
            out.append(rng.choice(FILLER_WORDS))
    return out

def heuristic_label(row: Dict[str, float], rng: random.Random) -> int:
    """
    Documented synthetic labeling rule: flags a sample as "elevated risk
    signal" if it shows a combination of slow/halting pace, high word-error
    rate, high phoneme mismatch, or heavy pausing -- proxies loosely
    associated with reading-fluency difficulty in the literature, combined
    here purely for demonstration purposes, not clinical validation.
    """
    score = 0
    if row["wpm"] < 80:
        score += 1
    if row["word_error_rate"] > 0.35:
        score += 1
    if row["phoneme_mismatch_proxy"] > 0.3:
        score += 1
    if row["pause_ratio"] > 0.35:
        score += 1

    label = 1 if score >= 2 else 0
    if rng.random() < LABEL_NOISE_RATE:
        label = 1 - label
    return label

def main() -> None:
    rng = random.Random(SEED)
    transcripts = json.loads((REAL_AUDIO_DIR / "transcripts.json").read_text())

    rows: List[Dict[str, float]] = []
    for clip_id, transcript in transcripts.items():
        audio_path = REAL_AUDIO_DIR / f"{clip_id}.wav"
        base_acoustic = extract_acoustic_features(audio_path)
        duration = wav_info(audio_path)["duration_sec"]
        prompt_words = tokenize(clean_transcript(transcript))

        for _ in range(VARIANTS_PER_CLIP):
            spoken_words = perturb_words(prompt_words, rng)
            duration_factor = rng.uniform(0.6, 1.8)

            row = {
                "wpm": estimate_wpm(len(spoken_words), duration * duration_factor),
                "word_error_rate": word_error_rate(prompt_words, spoken_words),
                "phoneme_mismatch_proxy": phoneme_mismatch_rate(prompt_words, spoken_words),
                "spoken_word_count": float(len(spoken_words)),
                "prompt_word_count": float(len(prompt_words)),
            }
            for key, value in base_acoustic.items():
                jitter = rng.gauss(0, 0.05 * (abs(value) + 1e-6))
                row[key] = value + jitter
            row["pause_ratio"] = min(max(row["pause_ratio"], 0.0), 1.0)

            row["label"] = heuristic_label(row, rng)
            rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
