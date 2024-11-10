import json
import os

import numpy as np
import polars as pl

from vaxstats.analysis.forecast import detect_fever_hypothermia, run_analysis
from vaxstats.analysis.residual import add_residuals_col, get_residual_bounds
from vaxstats.analysis.stats import get_column_stat, get_column_stats
from vaxstats.analysis.timeframe import (
    add_hourly_thresholds,
    calculate_stats_by_timeframe,
)
from vaxstats.utils import get_baseline_df, str_to_datetime


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


def test_calculate_stats_by_hour(example_forecast_df_baseline):
    df = example_forecast_df_baseline
    hourly_stats = calculate_stats_by_timeframe(
        df,
        timeframe="hour",
        data_column="y",
        pred_column="y_hat",
        date_column="ds",
        start_from_first=False,
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


def test_calculate_stats_by_day(example_forecast_df_baseline):
    df = example_forecast_df_baseline
    stats = calculate_stats_by_timeframe(
        df,
        timeframe="day",
        data_column="y",
        pred_column="y_hat",
        date_column="ds",
        start_from_first=True,
    )

    assert "y_median" in stats.columns
    assert "y_hat_median" in stats.columns
    assert "start_time" in stats.columns
    assert "end_time" in stats.columns
    assert "data_points" in stats.columns

    # Check if start_time is always less than or equal to end_time
    assert (stats["start_time"] <= stats["end_time"]).all()

    # Check if data_points is always positive
    assert (stats["data_points"] > 0).all()

    temps_median = stats["y_median"].to_numpy()
    assert np.allclose(
        temps_median[:4],
        np.array([37.635, 37.707, 37.62, 37.501]),
    )
    assert np.allclose(temps_median[-1], np.array([37.2575]))


def test_calculate_thresholds(example_forecast_df):
    df = example_forecast_df
    df = add_residuals_col(df)
    df = str_to_datetime(df, date_column="ds", date_fmt="%Y-%m-%d %H:%M:%S")

    baseline_days = 7.0
    baseline_hours = 24 * baseline_days

    df_baseline = get_baseline_df(df, baseline=baseline_hours)

    residual_lower, residual_upper = get_residual_bounds(df_baseline)
    hourly_stats = calculate_stats_by_timeframe(
        df, timeframe="hour", data_column="y", pred_column="y_hat", date_column="ds"
    )
    hourly_stats = add_hourly_thresholds(hourly_stats, residual_lower, residual_upper)

    assert hourly_stats.shape[1] == 8
    assert "fever_threshold" in hourly_stats.columns
    assert "hypo_threshold" in hourly_stats.columns

    fever_threshold = hourly_stats["fever_threshold"].to_numpy()
    hypo_threshold = hourly_stats["hypo_threshold"].to_numpy()

    assert np.allclose(
        fever_threshold[:3], np.array([39.09025125, 39.22112064, 38.65119125])
    )
    assert np.allclose(
        hypo_threshold[:3], np.array([38.56102016, 38.69188955, 38.12196017])
    )


def test_detect_fever_hypothermia(example_forecast_df):
    df = example_forecast_df
    df = add_residuals_col(df)
    df = str_to_datetime(df, date_column="ds", date_fmt="%Y-%m-%d %H:%M:%S")

    baseline_days = 7.0
    baseline_hours = 24 * baseline_days

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
        fever_threshold[:3], np.array([39.09025125, 39.22112064, 38.65119125])
    )
    assert np.allclose(
        hypo_threshold[:3], np.array([38.56102016, 38.69188955, 38.12196017])
    )


def test_get_all_stats(example_forecast_df):
    df = example_forecast_df
    df = str_to_datetime(df, date_column="ds", date_fmt="%Y-%m-%d %H:%M:%S")
    df = add_residuals_col(df)

    baseline_days = 7.0
    baseline_hours = 24 * baseline_days

    results = run_analysis(
        df,
        baseline=baseline_hours,
        data_column="y",
        pred_column="y_hat",
        residual_column="residual",
    )
    assert np.allclose(results["duration"]["total_hours"], 693.210555)
    assert results["baseline"]["degrees_of_freedom"] == 663
    assert np.allclose(results["baseline"]["average_temp"], 37.6765)
    assert np.allclose(results["baseline"]["std_dev_temp"], 0.72782)
    assert np.allclose(results["baseline"]["residual_sum_squares"], 5.1582)
    assert np.allclose(results["residual"]["max_residual"], 2.70556)
    assert np.allclose(results["residual"]["residual_upper_bound"], 0.264615542)
    assert results["fever"]["duration"] == 266
    assert results["hypothermia"]["duration"] == 156


def test_get_all_stats_m9324(m9324_forecast_df, path_tmp):
    df = m9324_forecast_df
    df = str_to_datetime(df, date_column="ds", date_fmt="%Y-%m-%dT%H:%M:%S%.f")
    df = add_residuals_col(df)

    baseline_days = 3.0
    baseline_hours = 24 * baseline_days

    results = run_analysis(
        df,
        baseline=baseline_hours,
        data_column="y",
        pred_column="y_hat",
        residual_column="residual",
    )

    json_path = os.path.join(path_tmp, "m9324.json")
    with open(json_path, "w+", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    assert np.allclose(results["duration"]["total_hours"], 248.75)
    assert results["baseline"]["degrees_of_freedom"] == 288
    assert np.allclose(results["baseline"]["average_temp"], 37.75336)
    assert np.allclose(results["baseline"]["std_dev_temp"], 0.56774)
    assert np.allclose(results["baseline"]["residual_sum_squares"], 2.74906)
    assert np.allclose(results["residual"]["max_residual"], 3.21165)
    assert np.allclose(results["residual"]["residual_upper_bound"], 0.2931)
    assert results["fever"]["duration"] == 153
    assert results["hypothermia"]["duration"] == 8
