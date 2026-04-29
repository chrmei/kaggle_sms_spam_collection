from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

from features import DenseFeatureTransformer
from vectorizers import build_text_transformer

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class EncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """Wrap XGBoost with label encoding for string class labels."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model_: Optional[XGBClassifier] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: Any, y: Any) -> "EncodedXGBClassifier":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed; install it to use model_name='xgboost'")
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.model_ = XGBClassifier(**self.kwargs)
        self.model_.fit(X, y_encoded)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.model_ is None or self.label_encoder_ is None:
            raise ValueError("Model is not fitted yet.")
        y_encoded = self.model_.predict(X)
        return self.label_encoder_.inverse_transform(y_encoded.astype(int))

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.model_.predict_proba(X)


class SafeTruncatedSVD(BaseEstimator, TransformerMixin):
    """Use requested SVD components but cap safely for small feature spaces."""

    def __init__(self, n_components: int = 300, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.svd_: Optional[TruncatedSVD] = None
        self.n_components_used_: int = 0

    def fit(self, X: Any, y: Any = None) -> "SafeTruncatedSVD":
        n_features = int(X.shape[1])
        # TruncatedSVD requires n_components <= n_features; keep at least 1.
        safe_n = max(1, min(self.n_components, n_features))
        self.n_components_used_ = safe_n
        self.svd_ = TruncatedSVD(n_components=safe_n, random_state=self.random_state)
        self.svd_.fit(X, y)
        return self

    def transform(self, X: Any) -> Any:
        if self.svd_ is None:
            raise ValueError("SafeTruncatedSVD is not fitted yet.")
        return self.svd_.transform(X)


def build_estimator(
    model_name: str = "logistic_regression",
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> BaseEstimator:
    params = model_params or {}
    if model_name == "logistic_regression":
        defaults = {"max_iter": 3000, "class_weight": "balanced", "random_state": random_state}
        return LogisticRegression(**{**defaults, **params})
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed; install it to use model_name='xgboost'")
        defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": -1,
        }
        return EncodedXGBClassifier(**{**defaults, **params})
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_sparse_dense_pipeline(
    vectorizer_mode: str = "tfidf_word_char",
    model_name: str = "logistic_regression",
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    svd_components: int = 300,
) -> Pipeline:
    if svd_components < 1:
        raise ValueError("svd_components must be >= 1")
    text_transformer = Pipeline(
        steps=[
            ("vectorize", build_text_transformer(vectorizer_mode=vectorizer_mode)),
            ("svd", SafeTruncatedSVD(n_components=svd_components, random_state=random_state)),
        ]
    )
    features = ColumnTransformer(
        transformers=[
            ("sparse_text", text_transformer, ["clean_ph"]),
            (
                "dense",
                Pipeline(
                    steps=[
                        ("fe", DenseFeatureTransformer(raw_col="text", clean_ph_col="clean_ph")),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
                ["text", "clean_ph"],
            ),
        ],
        sparse_threshold=0.3,
    )
    clf = build_estimator(
        model_name=model_name,
        model_params=model_params,
        random_state=random_state,
    )
    return Pipeline(steps=[("features", features), ("clf", clf)])
