from pathlib import Path

import pandas as pd


def resolve_data_path(csv_path: str) -> Path:
    candidate = Path(csv_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    project_root = Path(__file__).resolve().parents[1]
    fallback = project_root / csv_path
    if fallback.exists():
        return fallback
    return candidate


def load_dataset(csv_path: str = "data/raw/spam.csv") -> pd.DataFrame:
    resolved_path = resolve_data_path(csv_path)
    df = pd.read_csv(resolved_path, encoding="latin1")
    df = df.rename(columns={"v1": "target", "v2": "text"})
    expected = {"text", "target"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df[["text", "target"]].copy()
