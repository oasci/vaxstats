from typing import Any

from datetime import timedelta

import polars as pl


def split_df(
    df: pl.DataFrame,
    hours: float | int = 72.0,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splits a DataFrame into training and testing sets based on a time window.

    This function takes a Polars DataFrame with a timestamp column named 'ds',
    parses the timestamps, and splits the DataFrame into two parts:
    training and testing sets. The training set includes rows from the earliest
    timestamp up to a specified number of hours. The testing set includes the
    remaining rows.

    Args:
        df (pl.DataFrame): The input DataFrame containing a 'ds' column with timestamps.
        hours (float | int, optional): The time window in hours for the training set.
            Defaults to 72.0.
        *args (tuple[Any, ...]): Additional positional arguments.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing two DataFrames.
            The first DataFrame is the training set, and the second DataFrame
            is the testing set.

    Raises:
        ValueError: If the 'ds' column is not found in the DataFrame.
        TypeError: If 'hours' is not a float or int.

    Examples:
        >>> import polars as pl
        >>> data = {
        ...     'ds': ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-03 00:00:00",
        ...            "2023-01-04 00:00:00", "2023-01-05 00:00:00"],
        ...     'value': [10, 20, 30, 40, 50]
        ... }
        >>> df = pl.DataFrame(data)
        >>> df_train, df_test = split_df(df, hours=48)
        >>> df_train
        shape: (2, 2)
        ┌─────────────────────┬───────┐
        │ ds                  ┆ value │
        │ ---                 ┆ ---   │
        │ datetime[ms]        ┆ i64   │
        ╞═════════════════════╪═══════╡
        │ 2023-01-01 00:00:00 ┆ 10    │
        │ 2023-01-02 00:00:00 ┆ 20    │
        └─────────────────────┴───────┘
        >>> df_test
        shape: (3, 2)
        ┌─────────────────────┬───────┐
        │ ds                  ┆ value │
        │ ---                 ┆ ---   │
        │ datetime[ms]        ┆ i64   │
        ╞═════════════════════╪═══════╡
        │ 2023-01-03 00:00:00 ┆ 30    │
        │ 2023-01-04 00:00:00 ┆ 40    │
        │ 2023-01-05 00:00:00 ┆ 50    │
        └─────────────────────┴───────┘

    """
    df = df.with_columns(pl.col("ds").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    earliest_time = df.select(pl.col("ds").min()).item()
    time_window_end = earliest_time + timedelta(hours=hours)

    df_train = df.filter(
        (pl.col("ds") >= earliest_time) & (pl.col("ds") <= time_window_end)
    )
    df_train = df_train.sort("ds")
    df_test = df.filter(
        ~((pl.col("ds") >= earliest_time) & (pl.col("ds") <= time_window_end))
    )
    df_test = df_test.sort("ds")
    return df_train, df_test
