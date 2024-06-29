from vaxstats.io import load_file
from vaxstats.utils import split_df


def test_split(path_example_prepped_csv):
    df = load_file(path_example_prepped_csv, file_type="csv")
    df_train, df_test = split_df(df=df, hours=24.0)

    assert df_train.shape == (95, 3)
    assert df_test.shape == (2_626, 3)
