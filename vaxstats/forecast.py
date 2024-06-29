from typing import Any

import argparse

import numpy as np
import polars as pl

from .io import load_file
from .utils import split_df


def run_forecasting(
    df: pl.DataFrame,
    sf_model: Any,
    baseline_hours: float | int = 72.0,
    sf_model_args: tuple[Any, ...] = tuple(),
    sf_model_kwargs: dict[str, Any] = dict(),
) -> pl.DataFrame:
    """
    Runs a forecasting model on a DataFrame, splitting it into training and
    testing sets, and adding the forecast results to the original DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame containing the data to forecast.
        sf_model (Any): The forecasting model class to be used.
        baseline_hours (float | int, optional): The time window in hours for
            the training set. Defaults to 72.0.
        sf_model_args (tuple[Any, ...], optional): Positional arguments to pass
            to the forecasting model constructor. Defaults to an empty tuple.
        sf_model_kwargs (dict[str, Any], optional): Keyword arguments to pass to
            the forecasting model constructor. Defaults to an empty dictionary.

    Returns:
        pl.DataFrame: The original DataFrame with an additional 'y_hat' column
            containing the forecasted values.

    Raises:
        ValueError: If the 'y' column is not found in the training DataFrame.
        TypeError: If 'baseline_hours' is not a float or int.

    Examples:
        >>> import polars as pl
        >>> from some_forecasting_library import SomeForecastingModel
        >>> data = {
        ...     'ds': ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-03 00:00:00",
        ...            "2023-01-04 00:00:00", "2023-01-05 00:00:00"],
        ...     'y': [10, 20, 30, 40, 50]
        ... }
        >>> df = pl.DataFrame(data)
        >>> df = run_forecasting(df, SomeForecastingModel, baseline_hours=48)
        >>> df
        shape: (5, 3)
        ┌─────────────────────┬─────┬───────┐
        │ ds                  ┆ y   ┆ y_hat │
        │ ---                 ┆ --- ┆ ---   │
        │ datetime[ms]        ┆ i64 ┆ f64   │
        ╞═════════════════════╪═════╪═══════╡
        │ 2023-01-01 00:00:00 ┆ 10  ┆ 10.1  │
        │ 2023-01-02 00:00:00 ┆ 20  ┆ 19.9  │
        │ 2023-01-03 00:00:00 ┆ 30  ┆ 29.8  │
        │ 2023-01-04 00:00:00 ┆ 40  ┆ 39.5  │
        │ 2023-01-05 00:00:00 ┆ 50  ┆ 49.2  │
        └─────────────────────┴─────┴───────┘
    """
    df_train, df_test = split_df(df, hours=baseline_hours)

    model = sf_model(*sf_model_args, **sf_model_kwargs)
    results = model.forecast(
        y=df_train["y"].to_numpy(), h=df_test.shape[0], fitted=True
    )
    y_hat = np.concatenate((results["fitted"], results["mean"]))
    df = df.with_columns([pl.Series("y_hat", y_hat)])
    return df


def cli_forecasting():
    """
    Command-line interface for running the forecasting model on a DataFrame.

    This function uses argparse to parse command-line arguments and executes
    the run_forecasting function with the specified parameters.
    """
    parser = argparse.ArgumentParser(description="Run forecasting on a DataFrame.")

    parser.add_argument("file_path", type=str, help="Path to the input data file.")
    parser.add_argument(
        "sf_model", type=str, help="The forecasting model class to be used."
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="csv",
        choices=["excel", "csv"],
        help="Type of the input file (excel or csv). Defaults to `csv`.",
    )
    parser.add_argument(
        "--baseline_hours",
        type=float,
        default=72.0,
        help="The time window in hours for the training set. Defaults to 72.0.",
    )
    parser.add_argument(
        "--sf_model_args",
        type=str,
        default="()",
        help="Positional arguments for the forecasting model constructor, in tuple format. Defaults to '()'.",
    )
    parser.add_argument(
        "--sf_model_kwargs",
        type=str,
        default="{}",
        help="Keyword arguments for the forecasting model constructor, in dictionary format. Defaults to '{}'.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.csv",
        help="Path to save the output DataFrame with forecasted values. Defaults to 'output.csv'.",
    )

    args = parser.parse_args()

    # Parse the sf_model_args and sf_model_kwargs
    sf_model_args = eval(args.sf_model_args)
    sf_model_kwargs = eval(args.sf_model_kwargs)

    # Load the input file
    df = load_file(args.file_path, args.file_type.lower())

    # Import the forecasting model class dynamically
    module_name, class_name = args.sf_model.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    sf_model = getattr(module, class_name)

    # Run forecasting
    df_result = run_forecasting(
        df,
        sf_model,
        baseline_hours=args.baseline_hours,
        sf_model_args=sf_model_args,
        sf_model_kwargs=sf_model_kwargs,
    )

    # Save the result
    df_result.write_csv(args.output_path)
