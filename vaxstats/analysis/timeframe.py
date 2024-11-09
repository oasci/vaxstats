from typing import Literal

import polars as pl


def calculate_stats_by_timeframe(
    df: pl.DataFrame,
    timeframe: Literal["hour", "day"],
    data_column: str = "y",
    pred_column: str = "y_hat",
    date_column: str = "ds",
    start_from_first: bool = True,
) -> pl.DataFrame:
    """
    Calculate statistics by specified timeframe, with an option to start from the first timestamp.

    Args:
        df: The input DataFrame.
        timeframe: Group statistics by timeframe - options are "hour" and "day".
        data_column: Name of column containing observed data.
        pred_column: Name of column containing predicted data.
        date_column: The name of the column with timestamps.
        start_from_first: If `True`, groups by the first timestamp's interval;
            otherwise, uses calendar-based intervals.

    Returns:
        A DataFrame with aggregated statistics by specified timeframe.
    """
    if start_from_first:
        # Calculate elapsed time since the first timestamp and divide by duration to create intervals
        stats_by_timeframe = df.group_by_dynamic(
            index_column=date_column,
            every=f"1{timeframe.lower()[0]}",
            closed="both",
            start_by="datapoint",
        ).agg(
            pl.col(data_column).median().alias("y_median"),
            pl.col(pred_column).median().alias("y_hat_median"),
            pl.col(date_column).min().alias("start_time"),
            pl.col(date_column).max().alias("end_time"),
            pl.col(data_column).count().alias("data_points"),
        )
    else:
        # Use calendar-based truncation with `truncate`
        stats_by_timeframe = (
            df.with_columns(
                pl.col(date_column).dt.truncate(f"1{timeframe[0]}").alias(timeframe)
            )
            .group_by(timeframe)
            .agg(
                pl.col(data_column).median().alias("y_median"),
                pl.col(pred_column).median().alias("y_hat_median"),
                pl.col(date_column).min().alias("start_time"),
                pl.col(date_column).max().alias("end_time"),
                pl.col(data_column).count().alias("data_points"),
            )
            .sort(timeframe)
        )

    return stats_by_timeframe


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
