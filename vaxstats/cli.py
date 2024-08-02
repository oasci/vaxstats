import argparse
import ast
import os
import sys

import yaml
from loguru import logger

from . import __version__, enable_logging
from .forecast import cli_forecast
from .io import cli_prep


def parse_args_kwargs(args_str, kwargs_str):
    logger.debug("Attempting to parse args and kwargs")
    try:
        args = ast.literal_eval(args_str)
        kwargs = ast.literal_eval(kwargs_str)
        if not isinstance(args, tuple):
            args = (args,) if args else ()
        if not isinstance(kwargs, dict):
            raise ValueError("kwargs must be a dictionary")
    except (SyntaxError, ValueError) as e:
        logger.error(f"Error parsing args or kwargs: {e}")
        sys.exit(1)
    return args, kwargs


def load_yaml_config(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML file: {e}")
        sys.exit(1)


def setup_logging(args):
    if args.vv:
        log_level = 0  # TRACE
    elif args.v:
        log_level = 10  # DEBUG
    else:
        log_level = 20  # INFO

    enable_logging(
        log_level,
        True,
        args.logfile,
        log_format="<level>{level: <8}</level> | {message}",
    )


def main():
    parser = argparse.ArgumentParser(description="VaxStats CLI")
    parser.add_argument("-v", action="store_true", help="More log verbosity")
    parser.add_argument("-vv", action="store_true", help="Even more log verbosity")
    parser.add_argument("--logfile", help="Specify a file to write logs to")
    parser.add_argument("--config", help="Path to YAML configuration file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Prep subcommand
    prep_parser = subparsers.add_parser("prep", help="Prepare data for analysis")
    prep_parser.add_argument("input_file", help="Input file to process")
    prep_parser.add_argument(
        "--date_idx", type=int, default=0, help="The index of the date column."
    )
    prep_parser.add_argument(
        "--time_idx", type=int, default=1, help="The index of the time column."
    )
    prep_parser.add_argument(
        "--y_idx", type=int, default=2, help="The index of the target variable column."
    )
    prep_parser.add_argument(
        "--output", type=str, default="output.csv", help="Name to save prepped file to."
    )

    # Forecast subcommand
    forecast_parser = subparsers.add_parser(
        "forecast", help="Perform forecasting on prepared data"
    )
    forecast_parser.add_argument(
        "file_path", type=str, help="Path to the input data file."
    )
    forecast_parser.add_argument(
        "sf_model", type=str, help="The forecasting model class to be used."
    )
    forecast_parser.add_argument(
        "--baseline_hours",
        type=float,
        default=72.0,
        help="The time window in hours for the training set. Defaults to 72.0.",
    )
    forecast_parser.add_argument(
        "--sf_model_args",
        type=str,
        default="()",
        help="Positional arguments for the forecasting model constructor. Use Python literal syntax, e.g., '(1, \"string\", [1, 2, 3])'",
    )
    forecast_parser.add_argument(
        "--sf_model_kwargs",
        type=str,
        default="{}",
        help='Keyword arguments for the forecasting model constructor. Use Python dict syntax, e.g., \'{"param1": 1, "param2": "value"}\'',
    )
    forecast_parser.add_argument(
        "--output_path",
        type=str,
        default="output.csv",
        help="Path to save the output DataFrame with forecasted values. Defaults to 'output.csv'.",
    )

    args = parser.parse_args()
    setup_logging(args)

    logger.info(f"VaxStats v{__version__} by OASCI <us@oasci.org>")

    if args.config:
        logger.info(f"User provided YAML config file at: `{args.config}`")
        if not os.path.exists(args.config):
            logger.critical("File does not exist")
            logger.critical("Exiting")
            sys.exit(1)
        logger.info("Loading YAML file")
        config = load_yaml_config(args.config)
        # Update args with config, prioritizing command-line arguments
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    if args.command == "prep":
        logger.info("User selected `prep` command")
        cli_prep(args)
    elif args.command == "forecast":
        logger.info("User selected `forecast` command")
        sf_model_args, sf_model_kwargs = parse_args_kwargs(
            args.sf_model_args, args.sf_model_kwargs
        )
        cli_forecast(args, sf_model_args, sf_model_kwargs)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
