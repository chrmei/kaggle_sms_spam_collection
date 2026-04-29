from preprocessing import add_clean_columns, preprocess_sms


def test_preprocessing_preserves_alnum_tokens() -> None:
    text = "Txt WIN100 now to 80085 for £50 cash"
    cleaned = preprocess_sms(text, use_placeholders=False, preprocessing_mode="none")
    assert "WIN100" in cleaned


def test_preprocessing_applies_placeholders() -> None:
    text = "Claim £50 at http://offer.example.com call 123456789"
    cleaned = preprocess_sms(text, use_placeholders=True, preprocessing_mode="none")
    assert "<MONEY>" in cleaned
    assert "<URL>" in cleaned
    assert "<PHONE>" in cleaned or "<NUM>" in cleaned


def test_add_clean_columns_outputs_expected_columns(canary_df) -> None:
    out = add_clean_columns(canary_df, preprocessing_mode="stem")
    assert "clean" in out.columns
    assert "clean_ph" in out.columns
    assert len(out["clean_ph"]) == len(canary_df)
