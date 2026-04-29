from typing import Any, Dict, Mapping, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from artifacts import save_json, save_metrics_history, save_pipeline
from cli import build_config_from_args, build_train_parser
from config import RunConfig
from data import load_dataset
from modeling import build_sparse_dense_pipeline
from preprocessing import add_clean_columns


def _resolve_model_df(
    config: RunConfig,
    df: pd.DataFrame,
    preprocessed_variants: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if preprocessed_variants and config.preprocessing_mode in preprocessed_variants:
        return preprocessed_variants[config.preprocessing_mode]
    return add_clean_columns(
        df,
        text_col="text",
        preprocessing_mode=config.preprocessing_mode,
    )


def _build_binary_confusion_payload(y_true_binary: pd.Series, y_pred_binary) -> Dict[str, Any]:
    matrix = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel().tolist()
    return {
        "labels": ["ham", "spam"],
        "matrix": matrix.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_with_config(
    config: RunConfig,
    df: pd.DataFrame | None = None,
    preprocessed_variants: Mapping[str, pd.DataFrame] | None = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    if df is None:
        df = load_dataset(config.data_path)
    model_df = _resolve_model_df(config=config, df=df, preprocessed_variants=preprocessed_variants)
    X = model_df[["text", "clean_ph"]]
    y = model_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    model = build_sparse_dense_pipeline(
        vectorizer_mode=config.vectorizer_mode,
        model_name=config.model_name,
        model_params=config.model_params,
        random_state=config.random_state,
        svd_components=config.svd_components,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    y_binary = (y_score >= config.threshold).astype(int)
    y_true_binary = (y_test == "spam").astype(int)
    confusion_payload = _build_binary_confusion_payload(y_true_binary=y_true_binary, y_pred_binary=y_binary)

    metrics = {
        "config": config.to_dict(),
        "model_name": config.model_name,
        "vectorizer_mode": config.vectorizer_mode,
        "preprocessing_mode": config.preprocessing_mode,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "spam_precision": float(precision_score(y_true_binary, y_binary, zero_division=0)),
        "spam_recall": float(recall_score(y_true_binary, y_binary, zero_division=0)),
        "spam_f1": float(f1_score(y_true_binary, y_binary, zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "pr_auc": float(average_precision_score(y_true_binary, y_score)),
        "threshold": float(config.threshold),
        "confusion_matrix_threshold": confusion_payload,
    }
    return model, metrics


def run_training(config: RunConfig) -> Dict[str, Any]:
    model, metrics = train_with_config(config=config)
    save_pipeline(config.output_dir, model)
    save_metrics_history(config.output_dir, metrics)
    save_json(config.output_dir, "config.json", config.to_dict())
    return metrics


if __name__ == "__main__":
    parser = build_train_parser()
    args = parser.parse_args()
    cfg = build_config_from_args(args)
    result = run_training(cfg)
    print(result)
