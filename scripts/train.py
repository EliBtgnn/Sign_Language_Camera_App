from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "landmarks.csv"
MODEL_OUT = ROOT / "models" / "gesture_classifier.joblib"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Collect data first with src/collect_data.py")

    df = pd.read_csv(DATA_PATH).dropna()
    if "label" not in df or df["label"].nunique() < 2:
        raise ValueError("Need at least two labels.")
    if df["label"].value_counts().min() < 2:
        raise ValueError("Every label needs at least two samples.")

    print("Samples per class:")
    print(df["label"].value_counts().to_string())
    print()

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1,
        class_weight="balanced", max_features="sqrt"
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print(f"Accuracy on held-out split: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, zero_division=0))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"Saved model: {MODEL_OUT}")


if __name__ == "__main__":
    main()
