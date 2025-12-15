from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .model_utils import save_model

def train_from_csv(csv_path: Path, model_out: Path) -> None:
    """
    Expected CSV columns (example):
    wpm, word_error_proxy, phoneme_mismatch_proxy, label
    label: 0/1 (risk signal present)
    """
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    save_model(model, model_out)
    print(f"Saved model to {model_out}")
