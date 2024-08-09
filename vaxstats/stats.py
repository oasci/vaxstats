import numpy as np
import polars as pl
from loguru import logger

from .utils import split_df


def get_column_mean(df: pl.DataFrame, column_name: str) -> float:
    return float(df.select(pl.mean(column_name)).item())


def get_column_min(df: pl.DataFrame, column_name: str) -> float:
    return float(df.select(pl.min(column_name)).item())


def get_column_max(df: pl.DataFrame, column_name: str) -> float:
    return float(df.select(pl.max(column_name)).item())


def get_column_std(df: pl.DataFrame, column_name: str) -> float:
    return float(df.select(pl.std(column_name)).item())


def get_column_stats(
    df: pl.DataFrame,
    true_column: str = "y",
    date_column: str = "ds",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
    baseline: float | int | None = None,
) -> dict[str, float]:
    logger.info("Computing baseline statistics")

    if baseline is not None:
        logger.debug("baseline is not None")
        logger.debug(f"Retrieving baseline DataFrame of {baseline} hours")
        df = split_df(df, hours=baseline, date_column=date_column, date_fmt=date_fmt)[0]
    data = {}
    data["mean"] = get_column_mean(df, column_name=true_column)
    data["min"] = get_column_min(df, column_name=true_column)
    data["max"] = get_column_max(df, column_name=true_column)
    data["std"] = get_column_std(df, column_name=true_column)
    return data


def add_residuals_col(
    df: pl.DataFrame,
    residual_name: str = "residual",
    true_column: str = "y",
    pred_column: str = "y_hat",
) -> pl.DataFrame:
    logger.info("Computing residuals.")
    return df.with_columns(
        (pl.col(true_column) - pl.col(pred_column)).alias(residual_name)
    )


def get_residual_sum_square(
    df: pl.DataFrame, residual_column: str = "residual"
) -> float:
    residuals = df.get_column(residual_column).to_numpy()
    rss: float = np.sum(residuals**2)
    return rss


def get_residual_bounds(
    df: pl.DataFrame,
    residual_column: str = "residual",
    date_column: str = "ds",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
    baseline: float | int | None = None,
) -> tuple[float, float]:
    if baseline is not None:
        logger.debug("baseline is not None")
        logger.debug(f"Retrieving baseline DataFrame of {baseline} hours")
        df = split_df(df, hours=baseline, date_column=date_column, date_fmt=date_fmt)[0]
    n_baseline_rows = df.shape[0]

    rss = get_residual_sum_square(df, residual_column=residual_column)

    rss_normed = rss / n_baseline_rows
    rss_upper = 3 * rss_normed ** (1 / 2)
    rss_lower = -rss_upper
    return rss_lower, rss_upper
