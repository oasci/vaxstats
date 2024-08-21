from typing import Any, Callable

import numpy as np
import polars as pl
from loguru import logger

from .utils import split_df


def get_column_stat(df: pl.DataFrame, column_name: str, stat_func: Callable) -> float:
    """
    Generic function to get a statistic for a column.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column_name (str): The name of the column to compute the statistic for.
        stat_func (Callable): The statistical function to apply (e.g., pl.mean, pl.min).

    Returns:
        float: The computed statistic for the specified column.
    """
    return float(df.select(stat_func(column_name)).item())


def get_column_mean(df: pl.DataFrame, column_name: str) -> float:
    """
    Get the mean value of a column in a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column_name (str): The name of the column to compute the mean for.

    Returns:
        float: The mean value of the specified column.
    """
    return get_column_stat(df, column_name, pl.mean)


def get_column_min(df: pl.DataFrame, column_name: str) -> float:
    """
    Get the minimum value of a column in a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column_name (str): The name of the column to find the minimum for.

    Returns:
        float: The minimum value of the specified column.
    """
    return get_column_stat(df, column_name, pl.min)


def get_column_max(df: pl.DataFrame, column_name: str) -> float:
    """
    Get the maximum value of a column in a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column_name (str): The name of the column to find the maximum for.

    Returns:
        float: The maximum value of the specified column.
    """
    return get_column_stat(df, column_name, pl.max)


def get_column_std(df: pl.DataFrame, column_name: str) -> float:
    """
    Get the standard deviation of a column in a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column_name (str): The name of the column to compute the standard deviation for.

    Returns:
        float: The standard deviation of the specified column.
    """
    return get_column_stat(df, column_name, pl.std)


def get_baseline_df(
    df: pl.DataFrame, baseline: float | int | None, date_column: str, date_fmt: str
) -> pl.DataFrame:
    """
    Get baseline DataFrame if baseline is provided.

    Args:
        df (pl.DataFrame): The input DataFrame.
        baseline (float | int | None): The number of hours to use as baseline. If None, returns the original DataFrame.
        date_column (str): The name of the date column in the DataFrame.
        date_fmt (str): The date format string for parsing the date column.

    Returns:
        pl.DataFrame: The baseline DataFrame or the original DataFrame if baseline is None.
    """
    if baseline is not None:
        logger.debug(f"Retrieving baseline DataFrame of {baseline} hours")
        return split_df(df, hours=baseline, date_column=date_column, date_fmt=date_fmt)[
            0
        ]
    return df


def get_column_stats(
    df: pl.DataFrame,
    true_column: str = "y",
    date_column: str = "ds",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
    baseline: float | int | None = None,
) -> dict[str, float]:
    """
    Compute baseline statistics for a specified column.

    Args:
        df (pl.DataFrame): The input DataFrame.
        true_column (str, optional): The name of the column to compute statistics for. Defaults to "y".
        date_column (str, optional): The name of the date column. Defaults to "ds".
        date_fmt (str, optional): The date format string. Defaults to "%Y-%m-%d %H:%M:%S".
        baseline (float | int | None, optional): The number of hours to use as baseline. If None, uses the entire DataFrame. Defaults to None.

    Returns:
        dict[str, float]: A dictionary containing the mean, min, max, and standard deviation of the specified column.
    """
    logger.info("Computing baseline statistics")
    df = get_baseline_df(df, baseline, date_column, date_fmt)

    stats = ["mean", "min", "max", "std"]
    stat_funcs = [get_column_mean, get_column_min, get_column_max, get_column_std]

    return {stat: func(df, true_column) for stat, func in zip(stats, stat_funcs)}


def add_residuals_col(
    df: pl.DataFrame,
    residual_name: str = "residual",
    true_column: str = "y",
    pred_column: str = "y_hat",
) -> pl.DataFrame:
    """
    Add a residuals column to the DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        residual_name (str, optional): The name for the new residuals column. Defaults to "residual".
        true_column (str, optional): The name of the column with true values. Defaults to "y".
        pred_column (str, optional): The name of the column with predicted values. Defaults to "y_hat".

    Returns:
        pl.DataFrame: The DataFrame with the added residuals column.
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
        df (pl.DataFrame): The input DataFrame.
        residual_column (str, optional): The name of the residuals column. Defaults to "residual".

    Returns:
        float: The residual sum of squares.
    """
    residuals = df.get_column(residual_column).to_numpy()
    return float(np.sum(residuals**2))


def calculate_residual_bounds(rss: float, n_rows: int) -> tuple[float, float]:
    """
    Calculate upper and lower residual bounds.

    Args:
        rss (float): The residual sum of squares.
        n_rows (int): The number of rows in the DataFrame.

    Returns:
        tuple[float, float]: A tuple containing the lower and upper residual bounds.
    """
    rss_normed = rss / n_rows
    rss_upper = 3 * rss_normed ** (1 / 2)
    rss_lower = -rss_upper
    return rss_lower, rss_upper


def get_residual_bounds(
    df: pl.DataFrame,
    residual_column: str = "residual",
    date_column: str = "ds",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
    baseline: float | int | None = None,
) -> tuple[float, float]:
    """
    Calculate residual bounds for the DataFrame.

    Args:
        df: The input DataFrame.
        residual_column: The name of the residuals column. Defaults to "residual".
        date_column: The name of the date column. Defaults to "ds".
        date_fmt: The date format string. Defaults to "%Y-%m-%d %H:%M:%S".
        baseline: The number of hours to use as baseline. If None, uses the entire DataFrame. Defaults to None.

    Returns:
        tuple[float, float]: A tuple containing the lower and upper residual bounds.
    """
    df = get_baseline_df(df, baseline, date_column, date_fmt)
    n_baseline_rows = df.shape[0]
    rss = get_residual_sum_square(df, residual_column=residual_column)
    return calculate_residual_bounds(rss, n_baseline_rows)


def calculate_hourly_stats(
    df: pl.DataFrame, data_column: str, date_column: str = "ds"
) -> pl.DataFrame:
    """
    Calculate hourly median temperatures.

    Args:
        df: The input DataFrame.
        data_column: The name of the column to compute hourly stats with.
        date_column: The name of the column with timestamps. Defaults to "ds".

    Returns:
        A DataFrame with hourly median temperatures.
    """
    return (
        df.with_columns(
            pl.col(date_column)
            .str.strptime(pl.Datetime)
            .dt.truncate("1h")
            .alias("hour")
        )
        .groupby("hour")
        .agg(
            pl.col(data_column).median().alias("hourly_median_temp"),
            pl.col(date_column).min().alias("start_time"),
            pl.col(date_column).max().alias("end_time"),
            pl.col(data_column).count().alias("data_points"),
        )
        .sort("hour")
    )


def calculate_thresholds(
    hourly_stats: pl.DataFrame, residual_upper: float, residual_lower: float
) -> pl.DataFrame:
    """
    Calculate fever and hypothermia thresholds.

    Args:
        hourly_stats: The DataFrame with hourly median temperatures.
        residual_upper: The upper residual bound.
        residual_lower: The lower residual bound.

    Returns:
        A DataFrame with added columns for residual bounds and fever/hypothermia thresholds.
    """
    return hourly_stats.with_columns(
        [
            (pl.col("hourly_median_temp") + residual_upper).alias("fever_threshold"),
            (pl.col("hourly_median_temp") + residual_lower).alias("hypo_threshold"),
        ]
    )


def detect_fever_hypothermia(
    df: pl.DataFrame,
    pred_column: str = "y_hat",
    residual_column: str = "residual",
    date_column: str = "ds",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
    baseline: float | int | None = None,
) -> pl.DataFrame:
    """
    Main function to detect fever and hypothermia thresholds.

    This function orchestrates the process of calculating residual bounds,
    hourly statistics, and fever/hypothermia thresholds.

    Args:
        df: The input DataFrame.
        pred_column: The name of the column with predicted values. Defaults to "y_hat".
        residual_column: The name of the residuals column. Defaults to "residual".
        date_column: The name of the date column. Defaults to "ds".
        date_fmt: The date format string. Defaults to "%Y-%m-%d %H:%M:%S".
        baseline: The number of hours to use as baseline. If None, uses the entire DataFrame. Defaults to None.

    Returns:
        A DataFrame with hourly statistics and fever/hypothermia thresholds.
    """
    logger.info("Detecting fever and hypothermia thresholds")
    residual_lower, residual_upper = get_residual_bounds(
        df, residual_column, date_column, date_fmt, baseline=baseline
    )
    hourly_stats = calculate_hourly_stats(df, pred_column)
    return calculate_thresholds(hourly_stats, residual_upper, residual_lower)


def compute_stats_dict(
    df: pl.DataFrame,
    data_column: str,
    residual_column: str,
    hourly_stats: pl.DataFrame,
    residual_bounds: tuple[float, float],
    date_column: str = "ds",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> dict[str, Any]:
    """
    Compute statistics from the DataFrame and hourly stats, and return them as a dictionary.

    Args:
        df (pl.DataFrame): The original DataFrame with all data.
        hourly_stats (pl.DataFrame): The DataFrame with hourly statistics.
        residual_bounds (tuple[float, float]): The lower and upper residual bounds.

    Returns:
        Dict[str, Any]: A dictionary containing the computed statistics.
    """
    df = df.with_columns(pl.col("ds").str.strptime(pl.Datetime, date_fmt))

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
            df[date_column].max() - df[date_column].min()
        ).total_seconds()
        / 3600,
        "fever_hours": hourly_stats.filter(
            pl.col("hourly_median_temp") > pl.col("fever_threshold")
        ).shape[0],
        "hypothermia_hours": hourly_stats.filter(
            pl.col("hourly_median_temp") < pl.col("hypo_threshold")
        ).shape[0],
    }

    # Combine all statistics into a single dictionary
    stats_dict = {
        "baseline_stats": baseline_stats,
        "residual_stats": residual_stats,
        "duration_stats": duration_stats,
    }

    return stats_dict
