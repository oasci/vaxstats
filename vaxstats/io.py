from typing import Any, Literal

import polars as pl


def clean_df(df: pl.DataFrame, col_idx: int = 0) -> pl.DataFrame:
    """
    Cleans a DataFrame by dropping rows with null values in the specified column.

    Args:
        df (pl.DataFrame): The input DataFrame.
        col_idx (int, optional): The index of the column to check for null values.
            Defaults to 0.

    Returns:
        pl.DataFrame: A DataFrame with rows containing null values in the specified
            column removed.

    Raises:
        IndexError: If col_idx is out of range of the DataFrame's columns.

    Examples:
        >>> import polars as pl
        >>> data = {'a': [1, 2, None], 'b': [4, None, 6]}
        >>> df = pl.DataFrame(data)
        >>> clean_df(df)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ NaN │
        └─────┴─────┘
    """
    df = df.drop_nulls(subset=df.columns[col_idx])
    return df


def prep_forecast_df(
    df: pl.DataFrame,
    date_idx: int,
    time_idx: int,
    y_idx: int,
    time_fmt: str = "%m-%d-%y %I:%M:%S %p",
) -> pl.DataFrame:
    """
    Prepares a DataFrame for forecasting by combining date and time columns,
    and formatting them.

    Args:
        df (pl.DataFrame): The input DataFrame.
        date_idx (int): The index of the date column.
        time_idx (int): The index of the time column.
        y_idx (int): The index of the target variable column.
        time_fmt (str, optional): The format of the date and time strings.
            Defaults to "%m-%d-%y %I:%M:%S %p".

    Returns:
        pl.DataFrame: A DataFrame with a combined and formatted datetime column
            ready for forecasting.

    Raises:
        IndexError: If any of date_idx, time_idx, or y_idx are out of range of the
            DataFrame's columns.
        ValueError: If the date and time strings do not match the specified time_fmt.

    Examples:
        >>> import polars as pl
        >>> data = {'date': ["01-01-23", "01-02-23"], 'time': ["01:00:00 PM", "02:00:00 PM"], 'y': [10, 20]}
        >>> df = pl.DataFrame(data)
        >>> prep_forecast_df(df, date_idx=0, time_idx=1, y_idx=2)
        shape: (2, 3)
        ┌─────────────────────┬───────┬─────────────┐
        │ ds                  ┆ y     ┆ unique_id   │
        │ ---                 ┆ ---   ┆ ---         │
        │ str                 ┆ i64   ┆ i64         │
        ╞═════════════════════╪═══════╪═════════════╡
        │ 2023-01-01 13:00:00 ┆ 10    ┆ 0           │
        │ 2023-01-02 14:00:00 ┆ 20    ┆ 0           │
        └─────────────────────┴───────┴─────────────┘
    """
    # Get data into correct schema.
    df = df.select(df.columns[date_idx], df.columns[time_idx], df.columns[y_idx])
    # Preparing the elapsed time (i.e., ds in statsforecast).
    df = df.with_columns([pl.concat_str(df.columns[:2], separator=" ").alias("ds")])
    df = df.select(df.columns[-1], df.columns[-2])
    df.columns = ["ds", "y"]
    df = df.with_columns(pl.lit(0).alias("unique_id"))
    df = df.select(["unique_id"] + df.columns[:-1])

    # Parse and format datetime.
    df = df.with_columns(
        [
            pl.col("ds")
            .str.strptime(pl.Datetime, format=time_fmt)
            .alias("parsed_datetime")
        ]
    )
    df = df.with_columns(
        [
            pl.col("parsed_datetime")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .alias("formatted_datetime")
        ]
    )
    df = df.with_columns([pl.col("formatted_datetime").alias("ds")])
    df = df.drop(["parsed_datetime", "formatted_datetime"])

    df = df.drop_nulls()
    return df


def load_file(
    file_path: str,
    file_type: None | Literal["csv", "excel"] = None,
    *args: Any,
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Loads a file into a Polars DataFrame.

    Args:
        file_path (str): The path to the file.
        file_type (str, optional): The type of file to load. Supported types are 'excel' and 'csv'.
            Defaults to 'excel'.
        *args (Any): Additional positional arguments to pass to the Polars file reading function.
        **kwargs (Any): Additional keyword arguments to pass to the Polars file reading function.

    Returns:
        pl.DataFrame: The loaded DataFrame.

    Raises:
        TypeError: If the `file_type` is not supported.

    Examples:
        >>> import polars as pl
        >>> df = load_file("data.xlsx")
        >>> df = load_file("data.csv", file_type="csv")

    """
    if file_type is None:
        if ".xls" in file_path[-6:]:
            df: pl.DataFrame = pl.read_excel(file_path, *args, **kwargs)
        elif ".csv" in file_path[-4:]:
            df = pl.read_csv(file_path, *args, **kwargs)
        else:
            raise TypeError("Could not determine file type")
    else:
        if file_type == "excel":
            df = pl.read_excel(file_path, *args, **kwargs)
        elif file_type == "csv":
            df = pl.read_csv(file_path, *args, **kwargs)
        else:
            raise TypeError(f"{file_type} is not supported")
    return df


def cli_prep(args):
    df = load_file(args.input_file)
    df = clean_df(df)
    df = prep_forecast_df(df, args.date_idx, args.time_idx, args.y_idx)
    df.write_csv(args.output)
