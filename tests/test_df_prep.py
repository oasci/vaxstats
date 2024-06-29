from vaxstats.io import clean_df, load_file, prep_forecast_df


def test_load_excel(path_example_excel):
    df = load_file(path_example_excel, "excel")
    assert df.shape == (2_742, 10)


def test_clean_df(path_example_excel):
    df = load_file(path_example_excel, "excel")
    df = clean_df(df)
    assert df.shape == (2_742, 10)


def test_prep_df(path_example_excel):
    df = load_file(path_example_excel, "excel")
    df = clean_df(df)
    df = prep_forecast_df(df, 0, 1, 6)
    assert df.columns == ["unique_id", "ds", "y"]
    assert df.shape == (2_721, 3)
