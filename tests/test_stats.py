import numpy as np

from vaxstats.io import load_file
from vaxstats.stats import add_residuals_col, get_column_stats
from vaxstats.utils import split_df


def test_baseline_stats(path_example_forecast_csv):
    df = load_file(path_example_forecast_csv, file_type="csv")

    baseline_days = 7.0
    baseline_hours = 24 * baseline_days

    baseline_stats = get_column_stats(df, baseline=baseline_hours)

    # R code provides 37.6796, but I think this is a bug because
    # they do not drop blank rows.
    assert np.allclose(baseline_stats["mean"], 37.6765, atol=0.0001)
    assert np.allclose(baseline_stats["min"], 36.549, atol=0.001)
    assert np.allclose(baseline_stats["max"], 39.118, atol=0.001)
    assert np.allclose(baseline_stats["std"], 0.7278, atol=0.0001)


def test_residual_stats(path_example_forecast_csv):
    df = load_file(path_example_forecast_csv, file_type="csv")

    baseline_days = 7.0
    baseline_hours = 24 * baseline_days
    df = split_df(df, hours=baseline_hours)[0]

    df = add_residuals_col(df)
    residuals = df.get_column("residual").to_numpy()
    rss = np.sum(residuals**2)

    assert np.allclose(rss, 5.1582, atol=0.0001)
