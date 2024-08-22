import polars as pl


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
