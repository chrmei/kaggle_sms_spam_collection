from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD


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
    parser.add_argument("--svd-components", type=int, default=100)
    args = parser.parse_args()

    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("svd", TruncatedSVD(n_components=args.svd_components, random_state=42)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Saved model to: {model_path}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")


if __name__ == "__main__":
    main()
