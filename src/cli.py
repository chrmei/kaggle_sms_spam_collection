import argparse
from dataclasses import replace
from typing import Any

from config import RunConfig


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--preprocessing-mode",
        type=str,
        choices=["none", "lemmatize", "stem", "lemma_then_stem"],
        default=None,
    )
    parser.add_argument(
        "--vectorizer-mode",
        type=str,
        choices=["tfidf_word_char", "count_word_char", "tfidf_plus_count"],
        default=None,
    )
    parser.add_argument("--svd-components", type=int, default=None)
    parser.add_argument(
        "--model-name",
        type=str,
        choices=["logistic_regression", "xgboost"],
        default=None,
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    return parser


def build_config_from_args(args: argparse.Namespace) -> RunConfig:
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
    if args.preprocessing_mode is not None:
        updates["preprocessing_mode"] = args.preprocessing_mode
    if args.vectorizer_mode is not None:
        updates["vectorizer_mode"] = args.vectorizer_mode
    if args.svd_components is not None:
        updates["svd_components"] = args.svd_components
    if args.model_name is not None:
        updates["model_name"] = args.model_name
    if args.threshold is not None:
        updates["threshold"] = args.threshold
    if args.min_precision is not None:
        updates["min_precision_for_recommendation"] = args.min_precision
    return replace(config, **updates) if updates else config


def build_train_parser() -> argparse.ArgumentParser:
    return _base_parser("Train SMS spam model with config overrides.")


def build_evaluate_parser() -> argparse.ArgumentParser:
    return _base_parser("Evaluate SMS spam model with config overrides.")
