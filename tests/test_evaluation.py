import numpy as np

from evaluate import _build_confusion_df, _threshold_report, run_engineered_pipeline_eval


def test_confusion_df_shape() -> None:
    y_true = ["ham", "spam", "ham", "spam"]
    y_pred = ["ham", "spam", "spam", "spam"]
    matrix = _build_confusion_df(y_true, y_pred)
    assert matrix.shape == (2, 2)
    assert matrix.values.sum() == len(y_true)


def test_threshold_report_returns_recommendation() -> None:
    y_true_binary = np.array([0, 1, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.2, 0.75, 0.6, 0.3])
    report, recommendation = _threshold_report(y_true_binary, y_score, min_precision=0.8)
    assert not report.empty
    assert 0.0 <= recommendation["recommended_threshold"] <= 1.0


def test_run_engineered_eval_returns_spam_metrics(canary_df, run_config) -> None:
    metrics = run_engineered_pipeline_eval(canary_df, run_config)
    assert "spam_precision" in metrics
    assert "spam_recall" in metrics
    assert "pr_auc" in metrics
