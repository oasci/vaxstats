import numpy as np

from vaxstats.io import load_file
from vaxstats.utils import datetime_to_float, split_df, str_to_datetime


def test_split(path_example_prepped_csv):
    df = load_file(path_example_prepped_csv, file_type="csv")
    df = str_to_datetime(df, date_column="ds", date_fmt="%Y-%m-%d %H:%M:%S")
    df_train, df_test = split_df(df=df, hours=24.0)

    assert df_train.shape == (94, 3)
    assert df_test.shape == (2_627, 3)


def test_date_to_hours(path_example_prepped_csv):
    df = load_file(path_example_prepped_csv, file_type="csv")
    hours = datetime_to_float(df, time_unit="hours")

    assert np.allclose(hours[0], 0)
    assert np.allclose(hours[1], 0.25)
    assert np.allclose(hours[-1], 693.210556)
