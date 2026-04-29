from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class RunConfig:
    data_path: str = "data/raw/spam.csv"
    output_dir: str = "artifacts/latest"
    random_state: int = 42
    test_size: float = 0.2
    preprocessing_mode: str = "none"
    use_placeholders: bool = True
    vectorizer_mode: str = "tfidf_word_char"
    svd_components: int = 300
    model_name: str = "logistic_regression"
    threshold: float = 0.5
    min_precision_for_recommendation: float = 0.95
    model_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**payload)
