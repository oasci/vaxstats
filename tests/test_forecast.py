from statsforecast.models import ARIMA

from vaxstats.forecast import run_forecasting
from vaxstats.io import load_file


def test_arima(path_example_prepped_csv, path_example_forecast_csv):
    model_kwargs = {
        "order": (0, 0, 10),
        "seasonal_order": (0, 1, 1),
        "season_length": 96,
        "method": "CSS-ML",  # CSS-ML, ML, CSS
    }
    df = load_file(path_example_prepped_csv, "csv")
    baseline_days = 7.0
    baseline_hours = 24 * baseline_days
    df = run_forecasting(
        df,
        baseline_hours=baseline_hours,
        sf_model=ARIMA,
        sf_model_kwargs=model_kwargs,
    )
    df.write_csv(file=path_example_forecast_csv)
