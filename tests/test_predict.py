from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.model.model_utils import save_model
from src.model.predict import predict_risk


def _toy_model_path(tmp_path: Path) -> Path:
    X = pd.DataFrame({
        "wpm": [60, 150, 55, 160],
        "word_error_rate": [0.6, 0.05, 0.7, 0.0],
    })
    y = [1, 0, 1, 0]
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X, y)
    path = tmp_path / "toy_model.joblib"
    save_model(model, path)
    return path

def test_predict_risk_returns_plain_language_assessment(tmp_path: Path):
    model_path = _toy_model_path(tmp_path)
    result = predict_risk(model_path, {"wpm": 58.0, "word_error_rate": 0.65})

    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_band"] in {"low", "moderate", "elevated"}
    assert "disclaimer" in result

def test_predict_risk_handles_missing_features(tmp_path: Path):
    model_path = _toy_model_path(tmp_path)
    # Only provide one of the two features the model was trained on.
    result = predict_risk(model_path, {"wpm": 60.0})

    assert 0.0 <= result["risk_score"] <= 1.0
    assert "notes" in result
