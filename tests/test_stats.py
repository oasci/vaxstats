import numpy as np
import polars as pl

from vaxstats.stats import (
    add_residuals_col,
    calculate_hourly_stats,
    calculate_thresholds,
    compute_stats_dict,
    detect_fever_hypothermia,
    get_column_stat,
    get_column_stats,
    get_residual_bounds,
)
from vaxstats.utils import split_df


def test_get_column_stat(example_forecast_df):
    result = get_column_stat(example_forecast_df, "y", pl.mean)
    assert np.isclose(result, 37.89085, atol=0.0001)


def test_baseline_stats(example_forecast_df):
    df = example_forecast_df

    baseline_days = 7.0
    baseline_hours = 24 * baseline_days

    baseline_stats = get_column_stats(df, baseline=baseline_hours)

    # R code provides 37.6796, but I think this is a bug because
    # they do not drop blank rows.
    assert np.allclose(baseline_stats["mean"], 37.6765, atol=0.0001)
    assert np.allclose(baseline_stats["min"], 36.549, atol=0.001)
    assert np.allclose(baseline_stats["max"], 39.118, atol=0.001)
    assert np.allclose(baseline_stats["std"], 0.7278, atol=0.0001)


def test_residual_stats(example_forecast_df):
    df = example_forecast_df

    baseline_days = 7.0
    baseline_hours = 24 * baseline_days
    df = split_df(df, hours=baseline_hours)[0]

    df = add_residuals_col(df)
    residuals = df.get_column("residual").to_numpy()
    rss = np.sum(residuals**2)

    assert np.allclose(rss, 5.1582, atol=0.0001)


def test_residual_bounds(example_forecast_df):
    df = example_forecast_df
    df = add_residuals_col(df)
    baseline_days = 7.0
    baseline_hours = 24 * baseline_days
    residual_bounds = get_residual_bounds(df, baseline=baseline_hours)
    assert np.allclose(
        np.array(residual_bounds), np.array((-0.26462, 0.26462)), atol=0.0001
    )


def test_calculate_hourly_stats(example_forecast_df):
    result = calculate_hourly_stats(example_forecast_df, data_column="y")
    print(result)

    assert result.shape[0] > 0
    assert "hourly_median_temp" in result.columns
    assert "start_time" in result.columns
    assert "end_time" in result.columns
    assert "data_points" in result.columns

    # Check if the hours are unique and sorted
    assert result["hour"].is_unique().all()

    # Check if start_time is always less than or equal to end_time
    assert (result["start_time"] <= result["end_time"]).all()

    # Check if data_points is always positive
    assert (result["data_points"] > 0).all()

    temps_hourly_median = result["hourly_median_temp"].to_numpy()
    assert np.allclose(
        temps_hourly_median[:4], np.array([39.0015, 38.822, 38.784, 38.3575])
    )
    assert np.allclose(temps_hourly_median[-1], np.array([38.4795]))


def test_calculate_thresholds(example_forecast_df):
    example_forecast_df = add_residuals_col(example_forecast_df)
    hourly_stats = calculate_hourly_stats(example_forecast_df, data_column="y_hat")
    residual_lower, residual_upper = get_residual_bounds(example_forecast_df)
    result = calculate_thresholds(hourly_stats, residual_upper, residual_lower)

    assert result.shape[1] == 7
    assert "fever_threshold" in result.columns
    assert "hypo_threshold" in result.columns

    fever_threshold = result["fever_threshold"].to_numpy()
    hypo_threshold = result["hypo_threshold"].to_numpy()

    assert np.allclose(fever_threshold[:3], np.array([41.4767, 41.2973, 41.2594]))
    assert np.allclose(hypo_threshold[:3], np.array([36.4484, 36.2690, 36.2310]))


def test_detect_fever_hypothermia(example_forecast_df):
    example_forecast_df = add_residuals_col(example_forecast_df)
    result = detect_fever_hypothermia(example_forecast_df)
    print(result)
    assert result.shape[1] == 7  # Should have 5 columns
    assert "hourly_median_temp" in result.columns
    assert "fever_threshold" in result.columns
    assert "hypo_threshold" in result.columns


def test_get_all_stats(example_forecast_df):
    df = add_residuals_col(example_forecast_df)
    hourly_stats = detect_fever_hypothermia(df, pred_column="y_hat")
    residual_bounds = get_residual_bounds(df)
    print(hourly_stats)

    stats_dict = compute_stats_dict(
        df,
        data_column="y",
        residual_column="residual",
        hourly_stats=hourly_stats,
        residual_bounds=residual_bounds,
    )
    print(stats_dict)
    exit()
