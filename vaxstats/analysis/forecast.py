from typing import Any

import json

import polars as pl
from loguru import logger

from ..io import load_file
from ..utils import get_baseline_df, str_to_datetime
from .hourly import add_hourly_thresholds, calculate_hourly_stats
from .residual import add_residuals_col, get_residual_bounds, get_residual_sum_square
from .stats import get_column_max, get_column_mean, get_column_min, get_column_std


def detect_fever_hypothermia(
    df: pl.DataFrame,
    baseline: float | int,
    data_column: str = "y",
    pred_column: str = "y_hat",
    residual_column: str = "residual",
    date_column: str = "ds",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> tuple[pl.DataFrame, tuple[float, float]]:
    """
    Main function to detect fever and hypothermia thresholds.

    This function orchestrates the process of calculating baseline residual bounds,
    hourly statistics, and fever/hypothermia thresholds.

    Args:
        df: The input DataFrame.
        baseline: The number of hours to use as baseline.
        pred_column: The name of the column with predicted values.
        residual_column: The name of the residuals column.
        date_column: The name of the date column.
        date_fmt: The date format string.

    Returns:
        A DataFrame with hourly statistics and fever/hypothermia thresholds.

        Lower and upper residual bounds, respectively.
    """
    logger.info("Detecting fever and hypothermia thresholds")
    df_baseline = get_baseline_df(df, date_column, date_fmt, baseline)
    residual_bounds = get_residual_bounds(df_baseline, residual_column)
    hourly_stats = calculate_hourly_stats(df, data_column, pred_column, date_column)
    hourly_stats = add_hourly_thresholds(hourly_stats, *residual_bounds)
    return hourly_stats, residual_bounds


def run_analysis(
    df: pl.DataFrame,
    baseline: float | int,
    data_column: str = "y",
    pred_column: str = "y_hat",
    residual_column: str = "residual",
    date_column: str = "ds",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> dict[str, Any]:
    """
    Compute statistics from the DataFrame and hourly stats, and return them as a
    dictionary.

    Args:
        df: The original DataFrame with all data.
        baseline: The number of hours to use as baseline.
        data_column: Column name with observed data.
        pred_column: The name of the column with predicted values.
        residual_column: The name of the residuals column.
        date_column: The name of the date column.
        date_fmt: The date format string.

    Returns:
        A dictionary containing the computed statistics.
    """
    hourly_stats, residual_bounds = detect_fever_hypothermia(
        df,
        pred_column=pred_column,
        residual_column=residual_column,
        date_column=date_column,
        date_fmt=date_fmt,
        baseline=baseline,
    )

    df_baseline = get_baseline_df(df, date_column, date_fmt, baseline)

    # Compute baseline statistics
    baseline_stats = {
        "degrees_of_freedom": df_baseline.shape[0],
        "average_temp": float(get_column_mean(df_baseline, data_column)),
        "std_dev_temp": float(get_column_std(df_baseline, data_column)),
        "max_temp": float(get_column_max(df_baseline, data_column)),
        "min_temp": float(get_column_min(df_baseline, data_column)),
        "residual_sum_squares": float(get_residual_sum_square(df_baseline, residual_column)),
    }

    # Compute residual statistics
    residual_stats = {
        "max_residual": float(get_column_max(df, residual_column)),
        "residual_lower_bound": residual_bounds[0],
        "residual_upper_bound": residual_bounds[1],
    }

    # Compute duration statistics
    duration_stats = {
        "total_duration_hours": (
            df[date_column].max() - df[date_column].min()  # type: ignore
        ).total_seconds()
        / 3600,
        "fever_hours": hourly_stats.filter(
            pl.col("y_median") > pl.col("fever_threshold")
        ).shape[0],
        "hypothermia_hours": hourly_stats.filter(
            pl.col("y_median") < pl.col("hypo_threshold")
        ).shape[0],
    }

    # Combine all statistics into a single dictionary
    stats_dict = {
        "baseline": baseline_stats,
        "residual": residual_stats,
        "duration": duration_stats,
    }

    return stats_dict


def cli_analysis(args):
    df = load_file(args.file_path)
    df = str_to_datetime(df, date_column=args.date_column, date_fmt=args.datetime_fmt)
    df = add_residuals_col(df)

    results = run_analysis(
        df,
        baseline=args.baseline_hours,
        data_column=args.data_column,
        pred_column=args.pred_column,
        residual_column=args.residual_column,
        date_column=args.date_column,
        date_fmt=args.datetime_fmt,
    )

    # Save the result
    with open(args.output_path, "w+", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
