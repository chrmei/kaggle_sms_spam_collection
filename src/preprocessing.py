import html
import re
import unicodedata
from functools import lru_cache
from typing import Callable, Sequence

import nltk
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ---------- patterns ----------
URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|http://\S+)", flags=re.IGNORECASE)
MONEY_RE = re.compile(r"[£$€]\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|[£$€]\s*\d+(?:\.\d+)?")
PHONE_RE = re.compile(r"(?<!\w)\d(?:[\s-]?\d){6,}(?!\w)")
NUM_RE = re.compile(r"\b(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\b")
FRACTION_RE = re.compile(r"\d+\s*[⁄/]\s*\d+")
TOKEN_RE = re.compile(r"<[^>]+>|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", flags=re.UNICODE)
PREPROCESSING_MODES: tuple[str, ...] = ("none", "lemmatize", "stem", "lemma_then_stem")


def normalize_text_safe(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("å£", "£").replace("â£", "£").replace("Ì¼", "£")
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u00A0", " ")


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def apply_placeholders(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = URL_RE.sub(" <URL> ", text)
    text = MONEY_RE.sub(" <MONEY> ", text)
    text = PHONE_RE.sub(" <PHONE> ", text)
    text = FRACTION_RE.sub(" <FRACTION> ", text)
    text = NUM_RE.sub(" <NUM> ", text)
    return text


def _tokenize_for_morphology(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _stemmer() -> PorterStemmer:
    return PorterStemmer()


@lru_cache(maxsize=1)
def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)


def _morph_token(token: str, mode: str) -> str:
    if token.startswith("<") and token.endswith(">"):
        return token
    if not any(ch.isalpha() for ch in token):
        return token
    if mode == "lemmatize":
        return _lemmatizer().lemmatize(token.lower())
    if mode == "stem":
        return _stemmer().stem(token.lower())
    if mode == "lemma_then_stem":
        return _stemmer().stem(_lemmatizer().lemmatize(token.lower()))
    return token


def apply_morphology(text: str, mode: str = "none") -> str:
    if mode == "none":
        return text
    if mode not in PREPROCESSING_MODES:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")
    if mode in {"lemmatize", "lemma_then_stem"}:
        ensure_nltk_resources()
    tokens = _tokenize_for_morphology(text)
    transformed = [_morph_token(token, mode) for token in tokens]
    return clean_whitespace(" ".join(transformed))


def preprocess_sms(
    text: str,
    use_placeholders: bool = True,
    preprocessing_mode: str = "none",
) -> str:
    normalized = normalize_text_safe(text)
    with_placeholders = apply_placeholders(normalized) if use_placeholders else normalized
    morphed = apply_morphology(with_placeholders, preprocessing_mode)
    return clean_whitespace(morphed)


def add_clean_columns(
    df: pd.DataFrame,
    text_col: str = "text",
    preprocessing_mode: str = "none",
) -> pd.DataFrame:
    out = df.copy()
    series = out[text_col].astype(str)
    clean_fn: Callable[[str], str] = lambda x: preprocess_sms(
        x,
        use_placeholders=False,
        preprocessing_mode=preprocessing_mode,
    )
    clean_ph_fn: Callable[[str], str] = lambda x: preprocess_sms(
        x,
        use_placeholders=True,
        preprocessing_mode=preprocessing_mode,
    )
    out["clean"] = series.map(clean_fn)
    out["clean_ph"] = series.map(clean_ph_fn)
    return out


def build_preprocessed_variants(
    df: pd.DataFrame,
    text_col: str = "text",
    modes: Sequence[str] = PREPROCESSING_MODES,
) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    for mode in modes:
        variants[mode] = add_clean_columns(df=df, text_col=text_col, preprocessing_mode=mode)
    return variants
