import numpy as np
import polars as pl

from vaxstats.stats import (
    add_hourly_thresholds,
    add_residuals_col,
    calculate_hourly_stats,
    detect_fever_hypothermia,
    get_column_stat,
    get_column_stats,
    get_residual_bounds,
    run_analysis,
)
from vaxstats.utils import get_baseline_df


def test_get_column_stat(example_forecast_df):
    hourly_stats = get_column_stat(example_forecast_df, "y", pl.mean)
    assert np.isclose(hourly_stats, 37.89085, atol=0.0001)


def test_baseline_stats(example_forecast_df_baseline):
    baseline_stats = get_column_stats(example_forecast_df_baseline, column_name="y")

    # R code provides 37.6796, but I think this is a bug because
    # they do not drop blank rows.
    assert np.allclose(baseline_stats["mean"], 37.6765, atol=0.0001)
    assert np.allclose(baseline_stats["min"], 36.549, atol=0.001)
    assert np.allclose(baseline_stats["max"], 39.118, atol=0.001)
    assert np.allclose(baseline_stats["std"], 0.7278, atol=0.0001)


def test_residual_stats(example_forecast_df_baseline):
    df = example_forecast_df_baseline

    df = add_residuals_col(df)
    residuals = df.get_column("residual").to_numpy()
    rss = np.sum(residuals**2)

    assert np.allclose(rss, 5.1582, atol=0.0001)


def test_residual_bounds(example_forecast_df_baseline):
    df = example_forecast_df_baseline
    df = add_residuals_col(df)
    residual_bounds = get_residual_bounds(df)
    assert np.allclose(
        np.array(residual_bounds), np.array((-0.26462, 0.26462)), atol=0.0001
    )


def test_calculate_hourly_stats(example_forecast_df_baseline):
    hourly_stats = calculate_hourly_stats(
        example_forecast_df_baseline,
        data_column="y",
        pred_column="y_hat",
        date_column="ds",
    )

    assert "y_median" in hourly_stats.columns
    assert "y_hat_median" in hourly_stats.columns
    assert "start_time" in hourly_stats.columns
    assert "end_time" in hourly_stats.columns
    assert "data_points" in hourly_stats.columns

    # Check if the hours are unique and sorted
    assert hourly_stats["hour"].is_unique().all()

    # Check if start_time is always less than or equal to end_time
    assert (hourly_stats["start_time"] <= hourly_stats["end_time"]).all()

    # Check if data_points is always positive
    assert (hourly_stats["data_points"] > 0).all()

    temps_hourly_median = hourly_stats["y_hat_median"].to_numpy()
    assert np.allclose(
        temps_hourly_median[:4],
        np.array([38.9624986, 38.78317825, 38.74521667, 38.3191432]),
    )
    assert np.allclose(temps_hourly_median[-1], np.array([38.8151678]))


def test_calculate_thresholds(example_forecast_df, baseline_hours):
    df = example_forecast_df
    df = add_residuals_col(df)
    df_baseline = get_baseline_df(df, baseline=baseline_hours)

    residual_lower, residual_upper = get_residual_bounds(df_baseline)
    hourly_stats = calculate_hourly_stats(
        df, data_column="y", pred_column="y_hat", date_column="ds"
    )
    hourly_stats = add_hourly_thresholds(hourly_stats, residual_lower, residual_upper)

    assert hourly_stats.shape[1] == 8
    assert "fever_threshold" in hourly_stats.columns
    assert "hypo_threshold" in hourly_stats.columns

    fever_threshold = hourly_stats["fever_threshold"].to_numpy()
    hypo_threshold = hourly_stats["hypo_threshold"].to_numpy()

    assert np.allclose(
        fever_threshold[:3], np.array([39.22711414, 39.0477938, 39.00983221])
    )
    assert np.allclose(
        hypo_threshold[:3], np.array([38.69788305, 38.51856271, 38.48060112])
    )


def test_detect_fever_hypothermia(example_forecast_df, baseline_hours):
    df = example_forecast_df
    df = add_residuals_col(df)
    hourly_stats, residual_bounds = detect_fever_hypothermia(
        df, baseline=baseline_hours
    )

    assert hourly_stats.shape[1] == 8
    assert "y_hat_median" in hourly_stats.columns
    assert "fever_threshold" in hourly_stats.columns
    assert "hypo_threshold" in hourly_stats.columns

    fever_threshold = hourly_stats["fever_threshold"].to_numpy()
    hypo_threshold = hourly_stats["hypo_threshold"].to_numpy()

    assert np.allclose(
        fever_threshold[:3], np.array([39.22711414, 39.0477938, 39.00983221])
    )
    assert np.allclose(
        hypo_threshold[:3], np.array([38.69788305, 38.51856271, 38.48060112])
    )


def test_get_all_stats(example_forecast_df, baseline_hours):
    df = example_forecast_df
    df = add_residuals_col(df)

    results = run_analysis(
        df,
        baseline=baseline_hours,
        data_column="y",
        pred_column="y_hat",
        residual_column="residual",
    )
    assert results["baseline_stats"]["degrees_of_freedom"] == 2721
    assert np.allclose(results["baseline_stats"]["average_temp"], 37.89085)
    assert np.allclose(results["baseline_stats"]["std_dev_temp"], 0.70588941)
    assert np.allclose(results["baseline_stats"]["residual_sum_squares"], 1911.0984)
    assert np.allclose(results["residual_stats"]["max_residual"], 2.70556)
    assert np.allclose(results["residual_stats"]["residual_upper_bound"], 0.264615542)
    assert np.allclose(results["duration_stats"]["total_duration_hours"], 693.210555)
    assert results["duration_stats"]["fever_hours"] == 261
    assert results["duration_stats"]["hypothermia_hours"] == 157
