from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def _word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2)


def _char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)


def _word_count() -> CountVectorizer:
    return CountVectorizer(ngram_range=(1, 2), min_df=2)


def _char_count() -> CountVectorizer:
    return CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)


def build_text_transformer(vectorizer_mode: str) -> ColumnTransformer:
    if vectorizer_mode == "tfidf_word_char":
        return ColumnTransformer(
            transformers=[
                ("word_tfidf", _word_tfidf(), "clean_ph"),
                ("char_tfidf", _char_tfidf(), "clean_ph"),
            ],
            sparse_threshold=1.0,
        )

    if vectorizer_mode == "count_word_char":
        return ColumnTransformer(
            transformers=[
                ("word_count", _word_count(), "clean_ph"),
                ("char_count", _char_count(), "clean_ph"),
            ],
            sparse_threshold=1.0,
        )

    if vectorizer_mode == "tfidf_plus_count":
        return ColumnTransformer(
            transformers=[
                (
                    "hybrid",
                    FeatureUnion(
                        transformer_list=[
                            ("word_tfidf", _word_tfidf()),
                            ("char_tfidf", _char_tfidf()),
                            ("word_count", _word_count()),
                            ("char_count", _char_count()),
                        ]
                    ),
                    "clean_ph",
                )
            ],
            sparse_threshold=1.0,
        )

    raise ValueError(f"Unsupported vectorizer mode: {vectorizer_mode}")
