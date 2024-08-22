from typing import Any, Callable

import polars as pl
from loguru import logger


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
