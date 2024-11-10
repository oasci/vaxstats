import os

import pytest

from vaxstats import enable_logging
from vaxstats.io import load_file
from vaxstats.utils import get_baseline_df, str_to_datetime

TEST_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="session", autouse=True)
def turn_on_logging():
    enable_logging(10)


@pytest.fixture
def path_example_excel():
    return os.path.join(TEST_DIR, "files/example.xlsx")


@pytest.fixture
def path_m9324_excel():
    return os.path.join(TEST_DIR, "files/test-m9324.xlsx")


@pytest.fixture
def path_example_prepped_csv():
    return os.path.join(TEST_DIR, "files/example_prepped.csv")


@pytest.fixture
def path_m9324_prepped_csv():
    return os.path.join(TEST_DIR, "files/test-m9324.csv")


@pytest.fixture
def path_example_forecast_csv():
    return os.path.join(TEST_DIR, "files/example_forecast.csv")


@pytest.fixture
def path_forecast_csv_m9324():
    return os.path.join(TEST_DIR, "files/test-m9324-forecast.csv")


@pytest.fixture
def path_example_img():
    return os.path.join(TEST_DIR, "tmp/path_example_img.png")


@pytest.fixture
def path_tmp():
    return os.path.join(TEST_DIR, "tmp")


@pytest.fixture
def example_forecast_df(path_example_forecast_csv):
    df = load_file(path_example_forecast_csv, file_type="csv")
    return df


@pytest.fixture
def m9324_forecast_df(path_forecast_csv_m9324):
    df = load_file(path_forecast_csv_m9324, file_type="csv")
    return df


@pytest.fixture
def example_forecast_df_baseline(path_example_forecast_csv):
    baseline_days = 7.0
    baseline_hours = 24 * baseline_days

    df = load_file(path_example_forecast_csv, file_type="csv")
    df = str_to_datetime(df, date_column="ds", date_fmt="%Y-%m-%d %H:%M:%S")
    return get_baseline_df(df=df, date_column="ds", baseline=baseline_hours)
