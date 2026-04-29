import re
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing import add_clean_columns, preprocess_sms

ALNUM_TOKEN_RE = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d)\w+\b")
SHORTCODE_RE = re.compile(r"\b\d{4,6}\b")
REPEATED_PUNCT_RE = re.compile(r"([!?$£€])\1{1,}")

# lexicons
SPAM_TRIGGER_WORDS = {
    "free", "win", "winner", "claim", "urgent", "offer", "prize",
    "call", "txt", "text", "reply", "stop", "cash", "award",
}
CTA_PHRASES = (
    "call now",
    "click",
    "reply now",
    "text back",
    "claim now",
    "limited time",
    "last chance",
)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _count_placeholder(text: str, token: str) -> int:
    return text.count(token)


def _tokenize(text: str) -> List[str]:
    return [tok for tok in (text or "").split(" ") if tok]


def extract_dense_features(raw_text: str, clean_ph_text: str) -> Dict[str, float]:
    raw_text = raw_text or ""
    clean_ph_text = clean_ph_text or ""
    tokens = _tokenize(clean_ph_text)
    token_count = len(tokens)
    char_count = len(clean_ph_text)

    num_count = _count_placeholder(clean_ph_text, "<NUM>")
    url_count = _count_placeholder(clean_ph_text, "<URL>")
    money_count = _count_placeholder(clean_ph_text, "<MONEY>")
    phone_count = _count_placeholder(clean_ph_text, "<PHONE>")
    fraction_count = _count_placeholder(clean_ph_text, "<FRACTION>")
    placeholder_count = num_count + url_count + money_count + phone_count + fraction_count

    alpha_chars = sum(ch.isalpha() for ch in raw_text)
    upper_chars = sum(ch.isupper() for ch in raw_text)
    punct_bursts = len(REPEATED_PUNCT_RE.findall(raw_text))
    all_caps_tokens = sum(tok.isupper() and len(tok) >= 2 for tok in raw_text.split())
    mixed_alnum_tokens = len(ALNUM_TOKEN_RE.findall(raw_text))
    short_code_count = len(SHORTCODE_RE.findall(raw_text))

    lowered_tokens = [t.lower() for t in tokens]
    trigger_count = sum(1 for t in lowered_tokens if t in SPAM_TRIGGER_WORDS)
    unique_token_count = len(set(lowered_tokens))
    repeated_token_count = token_count - unique_token_count
    cta_hits = sum(1 for phrase in CTA_PHRASES if phrase in clean_ph_text.lower())

    first_window = " ".join(tokens[:5]).lower()
    last_window = " ".join(tokens[-5:]).lower() if tokens else ""

    features = {
        "char_len": float(char_count),
        "token_len": float(token_count),
        "avg_token_len": _safe_div(sum(len(t) for t in tokens), token_count),
        "placeholder_ratio": _safe_div(placeholder_count, token_count),
        "num_count": float(num_count),
        "url_count": float(url_count),
        "money_count": float(money_count),
        "phone_count": float(phone_count),
        "fraction_count": float(fraction_count),
        "has_url": float(url_count > 0),
        "has_money": float(money_count > 0),
        "has_phone": float(phone_count > 0),
        "uppercase_ratio": _safe_div(upper_chars, alpha_chars),
        "all_caps_token_count": float(all_caps_tokens),
        "punctuation_burst_count": float(punct_bursts),
        "mixed_alnum_count": float(mixed_alnum_tokens),
        "short_code_count": float(short_code_count),
        "trigger_word_count": float(trigger_count),
        "cta_phrase_count": float(cta_hits),
        "unique_token_ratio": _safe_div(unique_token_count, token_count),
        "repeated_token_ratio": _safe_div(repeated_token_count, token_count),
        "url_near_start": float("<url>" in first_window),
        "url_near_end": float("<url>" in last_window),
        "money_near_end": float("<money>" in last_window),
        "phone_near_end": float("<phone>" in last_window),
    }
    return features


def build_dense_feature_frame(
    raw_texts: Sequence[str],
    clean_ph_texts: Sequence[str],
) -> pd.DataFrame:
    rows = [
        extract_dense_features(raw_text=raw, clean_ph_text=clean_ph)
        for raw, clean_ph in zip(raw_texts, clean_ph_texts)
    ]
    return pd.DataFrame(rows, dtype=np.float64)


class DenseFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Build handcrafted dense features from raw text and clean_ph text.
    Expects a DataFrame with raw and clean_ph columns.
    """

    def __init__(self, raw_col: str = "text", clean_ph_col: str = "clean_ph") -> None:
        self.raw_col = raw_col
        self.clean_ph_col = clean_ph_col
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Iterable[str] = None) -> "DenseFeatureTransformer":
        dense_df = build_dense_feature_frame(X[self.raw_col], X[self.clean_ph_col])
        self.feature_names_ = dense_df.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        dense_df = build_dense_feature_frame(X[self.raw_col], X[self.clean_ph_col])
        return dense_df.to_numpy(dtype=np.float64)
