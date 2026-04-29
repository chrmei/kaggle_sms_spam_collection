from preprocessing import add_clean_columns
from modeling import build_sparse_dense_pipeline


def test_pipeline_fit_predict_and_proba(canary_df, run_config) -> None:
    frame = add_clean_columns(canary_df, preprocessing_mode=run_config.preprocessing_mode)
    X = frame[["text", "clean_ph"]]
    y = frame["target"]
    model = build_sparse_dense_pipeline(
        vectorizer_mode=run_config.vectorizer_mode,
        model_name=run_config.model_name,
        random_state=run_config.random_state,
    )
    model.fit(X, y)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert len(pred) == len(X)
    assert proba.shape[0] == len(X)
    assert proba.shape[1] == 2
