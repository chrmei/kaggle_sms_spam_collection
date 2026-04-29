import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("accuracy", "Accuracy"),
    ("spam_precision", "Spam Precision"),
    ("spam_recall", "Spam Recall"),
    ("spam_f1", "Spam F1"),
    ("weighted_f1", "Weighted F1"),
    ("pr_auc", "AUC (PR)"),
]

MODEL_KEYS = ("logistic_regression", "xgboost")
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create boxplots for XGBoost vs Logistic Regression from grid metrics."
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/grid1/metrics.json"),
        help="Path to metrics.json from the sweep run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/grid1/model_boxplots.png"),
        help="Output image path for the generated boxplots.",
    )
    return parser.parse_args()


def load_model_scores(metrics_path: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    scores = {
        model: {metric_key: [] for metric_key, _ in METRICS}
        for model in MODEL_KEYS
    }

    for model in MODEL_KEYS:
        entries = payload.get(model, {})
        for _, run_metrics in entries.items():
            for metric_key, _ in METRICS:
                if metric_key in run_metrics:
                    scores[model][metric_key].append(run_metrics[metric_key])

    return scores


def plot_boxplots(scores: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    model_order = list(MODEL_KEYS)
    labels = [MODEL_LABELS[model] for model in model_order]
    colors = ["#4C72B0", "#55A868"]

    for idx, (metric_key, metric_title) in enumerate(METRICS):
        axis = axes[idx]
        data = [scores[model][metric_key] for model in model_order]
        boxplot = axis.boxplot(data, patch_artist=True, tick_labels=labels)
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axis.set_title(metric_title)
        flat_values = [value for model_values in data for value in model_values]
        min_value = min(flat_values)
        max_value = max(flat_values)
        span = max(max_value - min_value, 1e-4)
        pad = span * 0.12
        lower = max(0.0, min_value - pad)
        upper = min(1.0, max_value + pad)
        if lower == upper:
            upper = min(1.0, lower + 1e-3)
        axis.set_ylim(lower, upper)
        axis.grid(axis="y", alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Grid1 Metrics: XGBoost vs Logistic Regression", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scores = load_model_scores(args.metrics_path)
    plot_boxplots(scores, args.output)
    print(f"Saved visualization to: {args.output}")
    for model in MODEL_KEYS:
        n_runs = len(scores[model]["accuracy"])
        print(f"{MODEL_LABELS[model]} runs: {n_runs}")


if __name__ == "__main__":
    main()
