from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import RunConfig


@pytest.fixture
def canary_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "tests" / "canary_sms.csv")


@pytest.fixture
def run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        data_path=str(ROOT / "tests" / "canary_sms.csv"),
        output_dir=str(tmp_path / "artifacts"),
        test_size=0.34,
        random_state=7,
        preprocessing_mode="none",
        vectorizer_mode="tfidf_word_char",
        model_name="logistic_regression",
        threshold=0.5,
        min_precision_for_recommendation=0.8,
    )
