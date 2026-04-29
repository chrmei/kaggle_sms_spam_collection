from pathlib import Path

import joblib
import pandas as pd

from train import run_training


def test_artifact_load_and_inference_contract(run_config) -> None:
    metrics = run_training(run_config)
    assert "spam_f1" in metrics

    model_path = Path(run_config.output_dir) / "model.joblib"
    assert model_path.exists()
    model = joblib.load(model_path)

    sample = pd.DataFrame([
        {"text": "WIN £1000 now", "clean_ph": "WIN <MONEY> now"},
        {"text": "See you at 5pm", "clean_ph": "See you at <NUM> pm"},
    ])
    pred = model.predict(sample)
    assert len(pred) == 2
