from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import numpy as np

from artifacts import save_dataframe, save_json, save_metrics_history
from cli import build_config_from_args, build_evaluate_parser
from config import RunConfig
from data import load_dataset
from modeling import build_sparse_dense_pipeline
from preprocessing import add_clean_columns

from pprint import pprint


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


def _compute_scores(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "spam_recall": float(recall_score(y_true, y_pred, pos_label="spam")),
        "spam_f1": float(f1_score(y_true, y_pred, pos_label="spam")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _build_confusion_df(y_true, y_pred) -> pd.DataFrame:
    labels = ["ham", "spam"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def _build_binary_confusion_payload(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> Dict[str, Any]:
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


def _threshold_report(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    min_precision: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_score)
    rows: List[Dict[str, float]] = []
    for idx, th in enumerate(thresholds):
        p = float(precision[idx + 1])
        r = float(recall[idx + 1])
        f1 = float((2 * p * r / (p + r)) if (p + r) else 0.0)
        rows.append({"threshold": float(th), "precision": p, "recall": r, "f1": f1})
    report = pd.DataFrame(rows)
    valid = report[report["precision"] >= min_precision]
    if valid.empty:
        best = report.sort_values("f1", ascending=False).head(1).iloc[0]
    else:
        best = valid.sort_values("recall", ascending=False).head(1).iloc[0]
    recommendation = {
        "recommended_threshold": float(best["threshold"]),
        "recommended_precision": float(best["precision"]),
        "recommended_recall": float(best["recall"]),
        "recommended_f1": float(best["f1"]),
        "min_precision_constraint": float(min_precision),
    }
    return report, recommendation


def run_engineered_pipeline_eval(
    df: pd.DataFrame,
    config: RunConfig,
    preprocessed_variants: Mapping[str, pd.DataFrame] | None = None,
) -> Dict[str, Any]:
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
    pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    y_true_binary = (y_test == "spam").astype(int).to_numpy()
    y_binary = (y_score >= config.threshold).astype(int)
    confusion_payload = _build_binary_confusion_payload(y_true_binary=y_true_binary, y_pred_binary=y_binary)
    metrics = {
        "config": config.to_dict(),
        "model_name": config.model_name,
        "vectorizer_mode": config.vectorizer_mode,
        "preprocessing_mode": config.preprocessing_mode,
        "accuracy": float(accuracy_score(y_test, pred)),
        "spam_precision": float(precision_score(y_true_binary, y_binary, zero_division=0)),
        "spam_recall": float(recall_score(y_true_binary, y_binary, zero_division=0)),
        "spam_f1": float(f1_score(y_true_binary, y_binary, zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "pr_auc": float(average_precision_score(y_true_binary, y_score)),
        "threshold": float(config.threshold),
        "confusion_matrix_threshold": confusion_payload,
    }
    threshold_df, recommendation = _threshold_report(
        y_true_binary=y_true_binary,
        y_score=y_score,
        min_precision=config.min_precision_for_recommendation,
    )
    save_dataframe(config.output_dir, "threshold_report.csv", threshold_df)
    save_dataframe(config.output_dir, "confusion_matrix.csv", _build_confusion_df(y_test, pred))
    save_json(config.output_dir, "threshold_recommendation.json", recommendation)
    return {**metrics, **recommendation}


def run_model_comparison(
    df: pd.DataFrame,
    config: RunConfig,
    preprocessed_variants: Mapping[str, pd.DataFrame] | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Optional[pd.DataFrame]]]:
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

    base_pipeline = build_sparse_dense_pipeline(
        vectorizer_mode=config.vectorizer_mode,
        model_name="logistic_regression",
        random_state=config.random_state,
        svd_components=config.svd_components,
    )
    comparisons: List[Dict[str, float]] = []
    confusion_matrices: Dict[str, Optional[pd.DataFrame]] = {}

    # Logistic Regression baseline.
    lr_pipeline = clone(base_pipeline)
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    comparisons.append({"model": "logistic_regression", **_compute_scores(y_test, lr_pred)})
    confusion_matrices["logistic_regression"] = _build_confusion_df(y_test, lr_pred)

    xgb_pipeline = clone(base_pipeline)
    try:
        xgb_pipeline = xgb_pipeline.set_params(
            clf=build_sparse_dense_pipeline(
                vectorizer_mode=config.vectorizer_mode,
                model_name="xgboost",
                random_state=config.random_state,
                svd_components=config.svd_components,
            ).named_steps["clf"]
        )
        xgb_pipeline.fit(X_train, y_train)
        xgb_pred = xgb_pipeline.predict(X_test)
        comparisons.append({"model": "xgboost", **_compute_scores(y_test, xgb_pred)})
        confusion_matrices["xgboost"] = _build_confusion_df(y_test, xgb_pred)
    except Exception:
        comparisons.append(
            {"model": "xgboost", "accuracy": float("nan"), "spam_recall": float("nan"), "spam_f1": float("nan"), "weighted_f1": float("nan")}
        )
        confusion_matrices["xgboost"] = None

    return comparisons, confusion_matrices


def run_vectorizer_benchmark(
    df: pd.DataFrame,
    config: RunConfig,
    preprocessed_variants: Mapping[str, pd.DataFrame] | None = None,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for mode in ("tfidf_word_char", "count_word_char", "tfidf_plus_count"):
        variant_config = RunConfig(**{**config.to_dict(), "vectorizer_mode": mode, "model_name": "logistic_regression"})
        metrics = run_engineered_pipeline_eval(df=df, config=variant_config, preprocessed_variants=preprocessed_variants)
        rows.append(
            {
                "vectorizer_mode": mode,
                "accuracy": metrics["accuracy"],
                "spam_precision": metrics["spam_precision"],
                "spam_recall": metrics["spam_recall"],
                "spam_f1": metrics["spam_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "pr_auc": metrics["pr_auc"],
            }
        )
    return rows


if __name__ == "__main__":
    parser = build_evaluate_parser()
    args = parser.parse_args()
    run_config = build_config_from_args(args)
    df_sms = load_dataset(run_config.data_path)
    eval_metrics = run_engineered_pipeline_eval(df_sms, run_config)
    print("Engineered pipeline score:")
    pprint(eval_metrics)

    comparison_rows, confusion_matrices = run_model_comparison(df_sms, run_config)
    comparison_df = pd.DataFrame(comparison_rows).set_index("model")
    print("\nModel comparison (engineered pipeline):")
    print(comparison_df.to_string(float_format=lambda x: f"{x:.4f}"))
    vectorizer_rows = run_vectorizer_benchmark(df_sms, run_config)
    print("\nVectorizer benchmark:")
    print(pd.DataFrame(vectorizer_rows).set_index("vectorizer_mode").to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nConfusion matrices:")
    for model_name, matrix_df in confusion_matrices.items():
        print(f"\n{model_name}:")
        if matrix_df is None:
            print("Not available (xgboost not installed).")
            continue
        print(matrix_df.to_string())
    save_metrics_history(run_config.output_dir, eval_metrics)
    save_json(run_config.output_dir, "config.json", run_config.to_dict())
    save_json(run_config.output_dir, "model_comparison.json", {"rows": comparison_rows})
    save_json(run_config.output_dir, "vectorizer_benchmark.json", {"rows": vectorizer_rows})
