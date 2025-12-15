from typing import Dict
import pandas as pd
from .model_utils import load_model

def predict_risk(model_path, features: Dict[str, float]) -> Dict[str, float]:
    model = load_model(model_path)
    X = pd.DataFrame([features])
    prob = getattr(model, "predict_proba", None)
    if prob is None:
        pred = float(model.predict(X)[0])
        return {"risk_score": pred}
    p = model.predict_proba(X)[0][1]
    return {"risk_score": float(p)}
