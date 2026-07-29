from pathlib import Path

import pandas as pd

from src.model.train import train_from_csv
from src.nlp.feature_extractor import extract_features


def test_extract_features_runs():
    prompt = ["the", "cat", "sat"]
    spoken = ["the", "cat"]
    res = extract_features(prompt, spoken, duration_sec=2.0)
    assert "wpm" in res.features
    assert "word_error_rate" in res.features
    assert "phoneme_mismatch_proxy" in res.features

def test_extract_features_perfect_match_has_zero_error():
    words = ["the", "cat", "sat"]
    res = extract_features(words, words, duration_sec=2.0)
    assert res.features["word_error_rate"] == 0.0

def test_train_from_csv_produces_model(tmp_path: Path):
    df = pd.DataFrame({
        "wpm": [60, 150, 55, 160, 50, 170],
        "word_error_rate": [0.6, 0.05, 0.7, 0.02, 0.65, 0.0],
        "label": [1, 0, 1, 0, 1, 0],
    })
    csv_path = tmp_path / "toy.csv"
    df.to_csv(csv_path, index=False)
    model_out = tmp_path / "toy_model.joblib"

    train_from_csv(csv_path, model_out)

    assert model_out.exists()
