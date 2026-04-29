import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from artifacts import save_json, save_metrics_history, save_pipeline
from config import RunConfig
from data import load_dataset
from preprocessing import build_preprocessed_variants
from train import train_with_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run grid sweep for SMS spam training.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--svd-components", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--preprocessing-modes", nargs="+", required=True)
    parser.add_argument("--vectorizer-modes", nargs="+", required=True)
    parser.add_argument("--model-names", nargs="+", required=True)
    return parser


def _build_base_config(args: argparse.Namespace) -> RunConfig:
    config = RunConfig.from_yaml(args.config) if args.config else RunConfig()
    updates: dict[str, Any] = {}
    if args.data_path is not None:
        updates["data_path"] = args.data_path
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.test_size is not None:
        updates["test_size"] = args.test_size
    if args.random_state is not None:
        updates["random_state"] = args.random_state
    if args.svd_components is not None:
        updates["svd_components"] = args.svd_components
    if args.threshold is not None:
        updates["threshold"] = args.threshold
    if args.min_precision is not None:
        updates["min_precision_for_recommendation"] = args.min_precision
    return replace(config, **updates) if updates else config


def _score(entry: dict[str, Any], metric_name: str) -> float:
    value = entry.get(metric_name, float("-inf"))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")


def _describe(entry: dict[str, Any]) -> str:
    cfg = entry.get("config", {})
    return (
        f"model={entry.get('model_name')} "
        f"prep={entry.get('preprocessing_mode')} "
        f"vec={entry.get('vectorizer_mode')} "
        f"svd={cfg.get('svd_components')} "
        f"recall={_score(entry, 'spam_recall'):.6f} "
        f"weighted_f1={_score(entry, 'weighted_f1'):.6f} "
        f"hash={entry.get('config_hash')}"
    )


def main() -> None:
    args = _build_parser().parse_args()
    base_config = _build_base_config(args)

    output_dir = Path(base_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.unlink(missing_ok=True)

    preprocessing_modes = tuple(args.preprocessing_modes)
    vectorizer_modes = tuple(args.vectorizer_modes)
    model_names = tuple(args.model_names)
    total_runs = len(preprocessing_modes) * len(vectorizer_modes) * len(model_names)

    df = load_dataset(base_config.data_path)
    variants = build_preprocessed_variants(df=df, text_col="text", modes=preprocessing_modes)

    run_idx = 0
    for prep in preprocessing_modes:
        for vectorizer in vectorizer_modes:
            for model_name in model_names:
                run_idx += 1
                run_config = replace(
                    base_config,
                    preprocessing_mode=prep,
                    vectorizer_mode=vectorizer,
                    model_name=model_name,
                )
                print(
                    f">>> run {run_idx}/{total_runs}: "
                    f"PREPROCESSING_MODE={prep} VECTORIZER_MODE={vectorizer} MODEL_NAME={model_name}"
                )
                model, metrics = train_with_config(
                    config=run_config,
                    df=df,
                    preprocessed_variants=variants,
                )
                save_pipeline(run_config.output_dir, model)
                save_metrics_history(run_config.output_dir, metrics)
                save_json(run_config.output_dir, "config.json", run_config.to_dict())

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = [
        entry
        for model_runs in payload.values()
        if isinstance(model_runs, dict)
        for entry in model_runs.values()
        if isinstance(entry, dict)
    ]
    if not rows:
        raise RuntimeError("No metrics entries found in metrics.json after sweep.")

    best_recall = max(rows, key=lambda x: (_score(x, "spam_recall"), _score(x, "weighted_f1")))
    best_weighted_f1 = max(rows, key=lambda x: (_score(x, "weighted_f1"), _score(x, "spam_recall")))
    print("")
    print(f"Best model by spam_recall: {_describe(best_recall)}")
    print(f"Best model by weighted_f1: {_describe(best_weighted_f1)}")


if __name__ == "__main__":
    main()
