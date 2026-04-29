import numpy as np

from features import build_dense_feature_frame, extract_dense_features


def test_extract_dense_features_contains_key_signals() -> None:
    row = extract_dense_features(
        raw_text="WIN £1000 NOW!!! text WIN100 to 80085",
        clean_ph_text="WIN <MONEY> NOW !!! text WIN100 to <PHONE>",
    )
    assert row["money_count"] >= 1
    assert row["phone_count"] >= 1
    assert row["mixed_alnum_count"] >= 1


def test_dense_feature_frame_shape(canary_df) -> None:
    raw = canary_df["text"].tolist()
    clean_ph = canary_df["text"].tolist()
    frame = build_dense_feature_frame(raw, clean_ph)
    assert frame.shape[0] == len(canary_df)
    assert frame.shape[1] > 5
    assert np.issubdtype(frame.dtypes.iloc[0], np.floating)
