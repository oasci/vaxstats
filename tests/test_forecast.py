import os

import numpy as np
from statsforecast.models import ARIMA

from vaxstats.cli import main
from vaxstats.forecast import run_forecasting
from vaxstats.io import load_file
from vaxstats.utils import str_to_datetime


def test_arima(path_example_prepped_csv):
    model_kwargs = {
        "order": (0, 0, 10),
        "seasonal_order": (0, 1, 1),
        "season_length": 96,
        "method": "CSS-ML",  # CSS-ML, ML, CSS
    }
    df = load_file(path_example_prepped_csv, "csv")
    df = str_to_datetime(df, date_column="ds", date_fmt="%Y-%m-%d %H:%M:%S")
    baseline_days = 7.0
    baseline_hours = 24 * baseline_days
    df = run_forecasting(
        df,
        baseline_hours=baseline_hours,
        sf_model=ARIMA,
        sf_model_kwargs=model_kwargs,
    )


def test_arima_cli(capsys, monkeypatch):
    output_path = "tests/tmp/test_arima_cli.csv"
    if os.path.exists(output_path):
        os.remove(output_path)
    cmd = [
        "vaxstats",
        "forecast",
        "tests/files/example_prepped.csv",
        "statsforecast.models.ARIMA",
        "--baseline_hours",
        "48",
        "--sf_model_args",
        "()",
        "--sf_model_kwargs",
        "{'order': (0, 0, 10), 'seasonal_order': (0, 1, 1), 'season_length': 96, 'method': 'CSS-ML'}",
        "--output_path",
        output_path,
    ]
    monkeypatch.setattr("sys.argv", cmd)
    main()
    captured = capsys.readouterr()
    print(captured)
    assert os.path.exists(output_path), "Output file was not created"

def test_arima_m3924(path_m3924_prepped_csv):
    model_kwargs = {
        "order": (0, 0, 10),
        "seasonal_order": (0, 1, 1),
        "season_length": 96,
        "method": "CSS-ML",  # CSS-ML, ML, CSS
    }
    df = load_file(path_m3924_prepped_csv, "csv")
    df = str_to_datetime(df, date_column="ds", date_fmt="%Y-%m-%dT%H:%M:%S%.f")
    baseline_days = 3.0
    baseline_hours = 24 * baseline_days
    df = run_forecasting(
        df,
        baseline_hours=baseline_hours,
        sf_model=ARIMA,
        sf_model_kwargs=model_kwargs,
    )
    assert np.allclose(df["y_hat"][0], 37.01, atol=0.01)
    assert np.allclose(df["y_hat"][-1], 38.13, atol=0.01)
