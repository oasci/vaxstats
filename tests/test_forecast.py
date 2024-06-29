from statsforecast.models import ARIMA

from vaxstats.forecast import run_forecasting
from vaxstats.io import load_file


def test_arima(path_example_prepped_csv):
    model_kwargs = {
        "order": (0, 0, 10),
        "seasonal_order": (0, 1, 1),
        "season_length": 96,
        "method": "ML",
    }
    df = load_file(path_example_prepped_csv, "csv")
    run_forecasting(
        df,
        baseline_hours=25.0,
        sf_model=ARIMA,
        sf_model_kwargs=model_kwargs,
    )
