PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTHONPATH := src
CONFIG :=
DATA_PATH := data/raw/spam.csv
OUTPUT_DIR := artifacts/latest
TEST_SIZE := 0.2
RANDOM_STATE := 42
PREPROCESSING_MODE := none
VECTORIZER_MODE := tfidf_word_char
SVD_COMPONENTS := 300
MODEL_NAME := logistic_regression
THRESHOLD := 0.5
MIN_PRECISION := 0.95
PREPROCESSING_MODES := none lemmatize stem lemma_then_stem
VECTORIZER_MODES := tfidf_word_char count_word_char tfidf_plus_count
MODEL_NAMES := logistic_regression xgboost

.PHONY: help install train eval test baseline sweep-all-combinations clean-artifacts clean-reports

help:
	@echo "Available targets:"
	@echo "  install         Install runtime dependencies"
	@echo "  train           Train model"
	@echo "                  vars:"
	@echo "                    CONFIG=<yaml_path|empty>"
	@echo "                    DATA_PATH=<csv_path>"
	@echo "                    OUTPUT_DIR=<dir_path>"
	@echo "                    TEST_SIZE=<float 0-1>"
	@echo "                    RANDOM_STATE=<int>"
	@echo "                    PREPROCESSING_MODE=<none|lemmatize|stem|lemma_then_stem>"
	@echo "                    VECTORIZER_MODE=<tfidf_word_char|count_word_char|tfidf_plus_count>"
	@echo "                    SVD_COMPONENTS=<int >= 1>"
	@echo "                    MODEL_NAME=<logistic_regression|xgboost>"
	@echo "                    THRESHOLD=<float 0-1>"
	@echo "                    MIN_PRECISION=<float 0-1>"
	@echo "  eval            Run evaluation + comparisons"
	@echo "                  vars:"
	@echo "                    CONFIG=<yaml_path|empty>"
	@echo "                    DATA_PATH=<csv_path>"
	@echo "                    OUTPUT_DIR=<dir_path>"
	@echo "                    TEST_SIZE=<float 0-1>"
	@echo "                    RANDOM_STATE=<int>"
	@echo "                    PREPROCESSING_MODE=<none|lemmatize|stem|lemma_then_stem>"
	@echo "                    VECTORIZER_MODE=<tfidf_word_char|count_word_char|tfidf_plus_count>"
	@echo "                    SVD_COMPONENTS=<int >= 1>"
	@echo "                    MODEL_NAME=<logistic_regression|xgboost>"
	@echo "                    THRESHOLD=<float 0-1>"
	@echo "                    MIN_PRECISION=<float 0-1>"
	@echo "  test            Run pytest suite"
	@echo "  baseline        Snapshot baseline metrics"
	@echo "                  vars: none (uses fixed defaults in command)"
	@echo "  sweep-all-combinations"
	@echo "                  Run all preprocessing/vectorizer/model combinations,"
	@echo "                  then print best by spam_recall and weighted_f1 from metrics.json"
	@echo "                  vars:"
	@echo "                    PREPROCESSING_MODES='none lemmatize stem lemma_then_stem'"
	@echo "                    VECTORIZER_MODES='tfidf_word_char count_word_char tfidf_plus_count'"
	@echo "                    MODEL_NAMES='logistic_regression xgboost'"
	@echo "                    OUTPUT_DIR=<dir_path>"
	@echo "  clean-artifacts Remove generated artifacts"
	@echo "  clean-reports   Remove generated reports"
	@echo ""
	@echo "Global config vars and defaults:"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  DATA_PATH=$(DATA_PATH)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  TEST_SIZE=$(TEST_SIZE)"
	@echo "  RANDOM_STATE=$(RANDOM_STATE)"
	@echo "  PREPROCESSING_MODE=$(PREPROCESSING_MODE)"
	@echo "  VECTORIZER_MODE=$(VECTORIZER_MODE)"
	@echo "  SVD_COMPONENTS=$(SVD_COMPONENTS)"
	@echo "  MODEL_NAME=$(MODEL_NAME)"
	@echo "  THRESHOLD=$(THRESHOLD)"
	@echo "  MIN_PRECISION=$(MIN_PRECISION)"
	@echo "Example:"
	@echo "  make train OUTPUT_DIR=artifacts/count_stem VECTORIZER_MODE=count_word_char PREPROCESSING_MODE=stem"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/train.py \
		$(if $(CONFIG),--config $(CONFIG),) \
		--data-path $(DATA_PATH) \
		--output-dir $(OUTPUT_DIR) \
		--test-size $(TEST_SIZE) \
		--random-state $(RANDOM_STATE) \
		--preprocessing-mode $(PREPROCESSING_MODE) \
		--vectorizer-mode $(VECTORIZER_MODE) \
		--svd-components $(SVD_COMPONENTS) \
		--model-name $(MODEL_NAME) \
		--threshold $(THRESHOLD) \
		--min-precision $(MIN_PRECISION)

eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/evaluate.py \
		$(if $(CONFIG),--config $(CONFIG),) \
		--data-path $(DATA_PATH) \
		--output-dir $(OUTPUT_DIR) \
		--test-size $(TEST_SIZE) \
		--random-state $(RANDOM_STATE) \
		--preprocessing-mode $(PREPROCESSING_MODE) \
		--vectorizer-mode $(VECTORIZER_MODE) \
		--svd-components $(SVD_COMPONENTS) \
		--model-name $(MODEL_NAME) \
		--threshold $(THRESHOLD) \
		--min-precision $(MIN_PRECISION)

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

baseline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c "import json; from pathlib import Path; from data import load_dataset; from evaluate import run_model_comparison, run_engineered_pipeline_eval; from config import RunConfig; df=load_dataset('data/raw/spam.csv'); cfg=RunConfig(); out={'engineered_logreg': run_engineered_pipeline_eval(df, cfg), 'model_comparison': run_model_comparison(df, cfg)[0]}; Path('reports/baseline').mkdir(parents=True, exist_ok=True); Path('reports/baseline/baseline_metrics.json').write_text(json.dumps(out, indent=2), encoding='utf-8'); print('reports/baseline/baseline_metrics.json')"

sweep-all-combinations:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/sweep.py \
		$(if $(CONFIG),--config $(CONFIG),) \
		--data-path $(DATA_PATH) \
		--output-dir $(OUTPUT_DIR) \
		--test-size $(TEST_SIZE) \
		--random-state $(RANDOM_STATE) \
		--svd-components $(SVD_COMPONENTS) \
		--threshold $(THRESHOLD) \
		--min-precision $(MIN_PRECISION) \
		--preprocessing-modes $(PREPROCESSING_MODES) \
		--vectorizer-modes $(VECTORIZER_MODES) \
		--model-names $(MODEL_NAMES)

clean-artifacts:
	rm -rf artifacts/latest

clean-reports:
	rm -rf reports/baseline
