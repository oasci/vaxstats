from typing import Any, Collection, Literal

from datetime import timedelta

import numpy as np
import numpy.typing as npt
import polars as pl
from loguru import logger


def split_df(
    df: pl.DataFrame,
    hours: float | int | Collection[float | int] = 72.0,
    date_column: str = "ds",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[pl.DataFrame, ...]:
    """
    Splits a DataFrame into multiple sets based on time window(s).

    This function takes a Polars DataFrame with a timestamp column named `ds`,
    parses the timestamps, and splits the DataFrame into $n$ parts.

    Args:
        df: The input DataFrame containing a 'ds' column with timestamps.
        hours: Time duration (exclusive bounds) to include in $n - 1$ splits. If this is
            `24`, then two splits with the following rows will be provided:

            -   Up to, but not including, `24` hours,
            -   Any rows at and after `24` hours.
        date_column: column name containing date information.
        date_fmt: [Date format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
            for date column.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing DataFrame splits.

    Raises:
        ValueError: If the 'ds' column is not found in the DataFrame.
        TypeError: If 'hours' is not a float or int.

     Examples:
        ```python
        >>> import polars as pl
        >>> data = {
        ...     'ds': ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-03 00:00:00",
        ...            "2023-01-04 00:00:00", "2023-01-05 00:00:00"],
        ...     'value': [10, 20, 30, 40, 50]
        ... }
        >>> df = pl.DataFrame(data)
        >>> splits = split_df(df, hours=[24, 48])
        >>> len(splits)
        3
        >>> splits[0]
        shape: (1, 2)
        ┌─────────────────────┬───────┐
        │ ds                  ┆ value │
        │ ---                 ┆ ---   │
        │ datetime[ms]        ┆ i64   │
        ╞═════════════════════╪═══════╡
        │ 2023-01-01 00:00:00 ┆ 10    │
        └─────────────────────┴───────┘
        >>> splits[1]
        shape: (2, 2)
        ┌─────────────────────┬───────┐
        │ ds                  ┆ value │
        │ ---                 ┆ ---   │
        │ datetime[ms]        ┆ i64   │
        ╞═════════════════════╪═══════╡
        │ 2023-01-02 00:00:00 ┆ 20    │
        │ 2023-01-03 00:00:00 ┆ 30    │
        └─────────────────────┴───────┘
        >>> splits[2]
        shape: (2, 2)
        ┌─────────────────────┬───────┐
        │ ds                  ┆ value │
        │ ---                 ┆ ---   │
        │ datetime[ms]        ┆ i64   │
        ╞═════════════════════╪═══════╡
        │ 2023-01-04 00:00:00 ┆ 40    │
        │ 2023-01-05 00:00:00 ┆ 50    │
        └─────────────────────┴───────┘
        ```
    """
    if isinstance(hours, (int, float)):
        hours = (hours,)
    logger.info(f"Splitting DataFrame into {len(hours) + 1}")

    df = df.with_columns(pl.col(date_column).str.strptime(pl.Datetime, date_fmt))
    earliest_time = df.select(pl.col(date_column).min()).item()
    logger.debug(f"Earliest time found: {earliest_time}")

    logger.debug("Sorting DataFrame by time")
    df = df.sort(date_column)

    splits = []
    previous_time = earliest_time
    for hour in hours:
        time_window_end = previous_time + timedelta(hours=hour)
        logger.debug(f"Finding lines between {previous_time} and {time_window_end}")
        df_split = df.filter(
            (pl.col(date_column) >= previous_time)
            & (pl.col(date_column) < time_window_end)
        )
        splits.append(df_split)
        previous_time = time_window_end

    # Handle the remaining data
    df_remaining = df.filter(pl.col(date_column) >= previous_time)
    splits.append(df_remaining)
    return tuple(splits)


def datetime_to_float(
    df: pl.DataFrame,
    time_unit: Literal["hours", "days"] = "days",
    date_column: str = "ds",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> npt.NDArray[np.float64]:
    dates = df.get_column(date_column)
    dates = dates.str.to_datetime(date_fmt)
    earliest_date = dates.min()
    if time_unit == "hours":
        time_factor = 1 / 3600  # seconds to hour
    elif time_unit == "days":
        time_factor = 1 / 86400  # seconds to day
    time = np.array(
        [(date - earliest_date).total_seconds() * time_factor for date in dates]
    )
    return time
