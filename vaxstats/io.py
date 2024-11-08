from typing import Any, Literal

import ast
import sys

import polars as pl
from loguru import logger


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
    logger.info("Cleaning DataFrame")
    df = df.drop_nulls(subset=df.columns[col_idx])
    return df


def prep_forecast_df(
    df: pl.DataFrame,
    date_idx: int,
    time_idx: int,
    y_idx: int,
    input_date_fmt: str = "%m-%d-%y",
    input_time_fmt: str = "%I:%M:%S %p",
    output_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> pl.DataFrame:
    """
    Prepares a DataFrame for forecasting by combining date and time columns,
    and formatting them.

    Args:
        df: The input DataFrame.
        date_idx: The index of the date column.
        time_idx: The index of the time column.
        y_idx: The index of the target variable column.
        input_date_fmt: The format of the input date strings. Defaults to "%m-%d-%y".
        input_time_fmt: The format of the input time strings. Defaults to "%I:%M:%S %p".
        output_fmt: The format of the output datetime strings.
            Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        A DataFrame with a combined and formatted datetime column ready for forecasting.

    Raises:
    IndexError: If any of date_idx, time_idx, or y_idx are out of range of the
        DataFrame's columns.
    ValueError: If the date and time strings do not match the specified formats.

    Notes:
        If `date_idx` and `time_idx` are the same, we combine `input_date_fmt` and
        `input_time_fmt` and load from the specified column.

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
    logger.info("Preparing DataFrame for statsforecast")

    # Validate column indices
    if max(date_idx, time_idx, y_idx) >= len(df.columns):
        raise IndexError("One or more column indices are out of range")

    # Select only the required columns using indices
    if date_idx == time_idx:
        df = df.select(df.columns[date_idx], df.columns[y_idx])
        df = df.rename({df.columns[0]: "ds"})
    else:
        df = df.select(df.columns[date_idx], df.columns[time_idx], df.columns[y_idx])
        logger.debug("Combining date and time columns")
        df = df.with_columns(
            [pl.concat_str([df.columns[0], df.columns[1]], separator=" ").alias("ds")]
        )
        logger.debug(
            f"Parsing datetimes with date format '{input_date_fmt}' and time format '{input_time_fmt}'"
        )
        df = df.with_columns(
            [
                pl.col("ds")
                .str.strptime(
                    pl.Datetime,
                    format=f"{input_date_fmt} {input_time_fmt}",
                    strict=False,
                )
                .alias("parsed_datetime")
            ]
        )

    logger.debug(f"Example row: {df[0]}")

    # Rename the y column
    df = df.rename({df.columns[1]: "y"})

    logger.debug("Adding unique_id column")
    df = df.with_columns(pl.lit(0).alias("unique_id"))
    df = df.select(["unique_id", "ds", "y"])
    logger.debug(f"Example row: {df[0]}")

    n_rows = df.shape[0]
    df = df.drop_nulls()
    logger.debug(f"Dropped {n_rows - df.shape[0]} rows containing at least one null")

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
    logger.info(f"Loading file from: `{file_path}`")
    if file_type is None:
        logger.debug("Attempt to match file type")
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


def _parse_args(args_str):
    logger.debug("Attempting to parse args")
    try:
        args = ast.literal_eval(args_str)
        if not isinstance(args, tuple):
            args = (args,) if args else ()
    except (SyntaxError, ValueError) as e:
        logger.error(f"Error parsing args: {e}")
        sys.exit(1)
    return args


def _parse_kwargs(kwargs_str):
    logger.debug("Attempting to parse kwargs")
    try:
        kwargs = ast.literal_eval(kwargs_str)
        if not isinstance(kwargs, dict):
            raise ValueError("kwargs must be a dictionary")
    except (SyntaxError, ValueError) as e:
        logger.error(f"Error parsing kwargs: {e}")
        sys.exit(1)
    return kwargs


def cli_peak(args):
    df = load_file(args.file_path)
    print(df.head(args.n))


def cli_prep(args):
    """
    Prepare data for analysis using command-line arguments.

    This function loads a file, cleans the DataFrame, prepares it for forecasting,
    and saves the result to a CSV file.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse.
            Expected attributes:
            - file_path (str): Path to the input file.
            - date_idx (int): Index of the date column.
            - time_idx (int): Index of the time column.
            - y_idx (int): Index of the target variable column.
            - input_date_fmt (str): Format of the input date strings.
            - input_time_fmt (str): Format of the input time strings.
            - output_fmt (str): Format of the output datetime strings.
            - output (str): Name of the output CSV file.

    Returns:
        None
    """
    df = load_file(args.file_path)
    df = clean_df(df)
    df = prep_forecast_df(
        df,
        date_idx=args.date_idx,
        time_idx=args.time_idx,
        y_idx=args.y_idx,
        input_date_fmt=args.input_date_fmt,
        input_time_fmt=args.input_time_fmt,
        output_fmt=args.output_fmt,
    )
    logger.info(f"Writing prepared CSV file to: `{args.output}`")
    df.write_csv(args.output)
