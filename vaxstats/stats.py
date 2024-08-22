from typing import Any, Callable

import json

import numpy as np
import polars as pl
from loguru import logger

from .io import load_file
from .utils import get_baseline_df, str_to_datetime


def get_column_stat(
    df: pl.DataFrame, column_name: str, stat_func: Callable[[str], Any]
) -> float:
    """
    Generic function to get a statistic for a column.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to compute the statistic for.
        stat_func: The statistical function to apply (e.g., pl.mean, pl.min).

    Returns:
        The computed statistic for the specified column.
    """
    return float(df.select(stat_func(column_name)).item())


def get_column_mean(df: pl.DataFrame, column_name: str) -> float:
    """
    Get the mean value of a column in a DataFrame.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to compute the mean for.

    Returns:
        The mean value of the specified column.
    """
    return get_column_stat(df, column_name, pl.mean)


def get_column_min(df: pl.DataFrame, column_name: str) -> float:
    """
    Get the minimum value of a column in a DataFrame.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to find the minimum for.

    Returns:
        The minimum value of the specified column.
    """
    return get_column_stat(df, column_name, pl.min)


def get_column_max(df: pl.DataFrame, column_name: str) -> float:
    """
    Get the maximum value of a column in a DataFrame.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to find the maximum for.

    Returns:
        The maximum value of the specified column.
    """
    return get_column_stat(df, column_name, pl.max)


def get_column_std(df: pl.DataFrame, column_name: str) -> float:
    """
    Get the standard deviation of a column in a DataFrame.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to compute the standard deviation for.

    Returns:
        The standard deviation of the specified column.
    """
    return get_column_stat(df, column_name, pl.std)


def get_column_stats(df: pl.DataFrame, column_name: str) -> dict[str, float]:
    """
    Compute baseline statistics for a specified column.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to compute statistics for.

    Returns:
        A dictionary containing the mean, min, max, and standard deviation of the
            specified column.
    """
    logger.info("Computing all column statistics")
    stats = ["mean", "min", "max", "std"]
    stat_funcs = [get_column_mean, get_column_min, get_column_max, get_column_std]

    return {stat: func(df, column_name) for stat, func in zip(stats, stat_funcs)}


def add_residuals_col(
    df: pl.DataFrame,
    residual_name: str = "residual",
    true_column: str = "y",
    pred_column: str = "y_hat",
) -> pl.DataFrame:
    """
    Add a residuals column to the DataFrame.

    Args:
        df: The input DataFrame.
        residual_name: The name for the new residuals column.
        true_column: The name of the column with true values.
        pred_column: The name of the column with predicted values.

    Returns:
        The DataFrame with the added residuals column.
    """
    logger.info("Computing residuals.")
    return df.with_columns(
        (pl.col(true_column) - pl.col(pred_column)).alias(residual_name)
    )


def get_residual_sum_square(
    df: pl.DataFrame, residual_column: str = "residual"
) -> float:
    """
    Calculate the residual sum of squares.

    Args:
        df: The input DataFrame.
        residual_column: The name of the residuals column.

    Returns:
        The residual sum of squares.
    """
    residuals = df.get_column(residual_column).to_numpy()
    return float(np.sum(residuals**2))


def calculate_residual_bounds(rss: float, n_rows: int) -> tuple[float, float]:
    """
    Calculate upper and lower residual bounds.

    Args:
        rss: The residual sum of squares.
        n_rows: The number of rows in the DataFrame.

    Returns:
        A tuple containing the lower and upper residual bounds.
    """
    rss_normed = rss / n_rows
    rss_upper = 3 * rss_normed ** (1 / 2)
    rss_lower = -rss_upper
    return rss_lower, rss_upper


def get_residual_bounds(
    df: pl.DataFrame,
    residual_column: str = "residual",
) -> tuple[float, float]:
    """
    Calculate residual bounds for the DataFrame.

    Args:
        df: The input DataFrame.
        residual_column: The name of the residuals column.

    Returns:
        A tuple containing the lower and upper residual bounds.
    """
    n_rows = df.shape[0]
    rss = get_residual_sum_square(df, residual_column=residual_column)
    return calculate_residual_bounds(rss, n_rows)


def calculate_hourly_stats(
    df: pl.DataFrame,
    data_column: str = "y",
    pred_column: str = "y_hat",
    date_column: str = "ds",
) -> pl.DataFrame:
    """
    Calculate hourly median temperatures.

    Args:
        df: The input DataFrame.
        data_column: Name of column containing observed data.
        pred_column: Name of column containing predicted data.
        date_column: The name of the column with timestamps.

    Returns:
        A DataFrame with hourly median temperatures.
    """
    return (
        df.with_columns(pl.col(date_column).dt.truncate("1h").alias("hour"))
        .groupby("hour")
        .agg(
            pl.col(data_column).median().alias("y_median"),
            pl.col(pred_column).median().alias("y_hat_median"),
            pl.col(date_column).min().alias("start_time"),
            pl.col(date_column).max().alias("end_time"),
            pl.col(data_column).count().alias("data_points"),
        )
        .sort("hour")
    )


def add_hourly_thresholds(
    hourly_stats: pl.DataFrame, residual_lower: float, residual_upper: float
) -> pl.DataFrame:
    """
    Calculate and add fever and hypothermia thresholds.

    Args:
        hourly_stats: The DataFrame with hourly median temperatures.
        residual_lower: The lower residual bound.
        residual_upper: The upper residual bound.

    Returns:
        A DataFrame with added columns for residual bounds and fever/hypothermia
        thresholds.
    """
    return hourly_stats.with_columns(
        [
            (pl.col("y_hat_median") + residual_lower).alias("hypo_threshold"),
            (pl.col("y_hat_median") + residual_upper).alias("fever_threshold"),
        ]
    )


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

    # Compute baseline statistics
    baseline_stats = {
        "degrees_of_freedom": df.shape[0],
        "average_temp": float(get_column_mean(df, data_column)),
        "std_dev_temp": float(get_column_std(df, data_column)),
        "max_temp": float(get_column_max(df, data_column)),
        "min_temp": float(get_column_min(df, data_column)),
        "residual_sum_squares": float(get_residual_sum_square(df, residual_column)),
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
        "baseline_stats": baseline_stats,
        "residual_stats": residual_stats,
        "duration_stats": duration_stats,
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
        json.dump(results, f)
