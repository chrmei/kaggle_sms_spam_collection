from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin1")
    if {"v1", "v2"}.issubset(df.columns):
        df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["text", "label"]].dropna()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw/spam.csv")
    parser.add_argument("--model-path", default="prototype/model.joblib")
    args = parser.parse_args()

    df = load_data(args.data_path)
    _, X_test, _, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = joblib.load(Path(args.model_path))
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix [ham, spam]:")
    print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Sample predictions vs actual:")
    sample = pd.DataFrame({"text": X_test.values, "actual": y_test.values, "predicted": y_pred})
    print(sample.head(10).to_string(index=False))

    print("\nPredictions on real messages:")
    real_messages = [
        "Hey, are we still meeting at 7 tonight?",
        "Congratulations! You won a free iPhone. Claim now!",
        "Please call me when you arrive.",
        "URGENT! Your account is suspended. Verify immediately. CLICK: www.google.com",
    ]
    real_preds = model.predict(real_messages)
    real_probs = model.predict_proba(real_messages)
    for msg, pred, prob in zip(real_messages, real_preds, real_probs):
        print(f"- {pred:4} | spam_prob={prob[1]:.4f} | {msg}")


if __name__ == "__main__":
    main()
