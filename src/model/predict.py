from typing import Any, Dict, List

import pandas as pd

from .model_utils import load_model

RISK_BANDS = (
    (0.33, "low", "Reading signals in this sample were mostly in line with the prompt."),
    (0.66, "moderate", "Some reading signals in this sample differed from the prompt "
                        "and may be worth a follow-up screening."),
    (1.01, "elevated", "Several reading signals in this sample differed noticeably from "
                        "the prompt. Consider a follow-up with a reading specialist."),
)

# Human-readable labels for the features a RandomForest might flag as most
# influential, so the explanation reads as plain language, not a variable name.
FEATURE_DESCRIPTIONS = {
    "wpm": "reading pace",
    "word_error_rate": "how closely the words spoken matched the prompt",
    "phoneme_mismatch_proxy": "how closely pronunciation matched the prompt",
    "pause_ratio": "amount of pausing/hesitation while reading",
}

def _risk_band(score: float):
    for threshold, band, message in RISK_BANDS:
        if score < threshold:
            return band, message
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]

def _top_contributing_features(model, feature_names: List[str], n: int = 2) -> List[str]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    ranked = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)
    described = [FEATURE_DESCRIPTIONS.get(name) for name, _ in ranked]
    return [d for d in described if d][:n]

def predict_risk(model_path, features: Dict[str, float]) -> Dict[str, Any]:
    """
    Run the trained classifier and translate its output into a plain-language
    risk assessment rather than a bare probability, per the project's
    human-centered design goal. Missing/extra feature columns (e.g. because
    acoustic feature extraction failed for a given clip) are aligned against
    what the model was actually trained on, defaulting missing ones to 0 and
    surfacing that in `notes` instead of failing the request.
    """
    model = load_model(model_path)
    expected = list(getattr(model, "feature_names_in_", features.keys()))

    row = {name: float(features.get(name, 0.0)) for name in expected}
    missing = [name for name in expected if name not in features]

    X = pd.DataFrame([row], columns=expected)

    predict_proba = getattr(model, "predict_proba", None)
    if predict_proba is None:
        score = float(model.predict(X)[0])
    else:
        score = float(predict_proba(X)[0][1])

    band, message = _risk_band(score)
    top_features = _top_contributing_features(model, expected)

    result: Dict[str, Any] = {
        "risk_score": score,
        "risk_band": band,
        "summary": message,
        "disclaimer": "Informational screening signal only, not a medical diagnosis.",
    }
    if top_features:
        result["contributing_factors"] = top_features
    if missing:
        result["notes"] = f"Defaulted missing features to 0: {', '.join(missing)}"
    return result
