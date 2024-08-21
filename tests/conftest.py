import os

import pytest

from vaxstats import enable_logging
from vaxstats.io import load_file

TEST_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="session", autouse=True)
def turn_on_logging():
    enable_logging(10)


@pytest.fixture
def path_example_excel():
    return os.path.join(TEST_DIR, "files/example.xlsx")


@pytest.fixture
def path_example_prepped_csv():
    return os.path.join(TEST_DIR, "files/example_prepped.csv")


@pytest.fixture
def path_example_forecast_csv():
    return os.path.join(TEST_DIR, "files/example_forecast.csv")


@pytest.fixture
def path_example_img():
    return os.path.join(TEST_DIR, "tmp/path_example_img.png")


@pytest.fixture
def example_forecast_df(path_example_forecast_csv):
    return load_file(path_example_forecast_csv, file_type="csv")
