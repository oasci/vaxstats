import numpy as np
import polars as pl
from loguru import logger


def add_residuals_col(
    df: pl.DataFrame,
    residual_name: str = "residual",
    true_column: str = "y",
    pred_column: str = "y_hat",
) -> pl.DataFrame:
    """
    Add a residuals column to the DataFrame.

    Args:
        df: The input DataFrame.
        residual_name: The name for the new residuals column.
        true_column: The name of the column with true values.
        pred_column: The name of the column with predicted values.

    Returns:
        The DataFrame with the added residuals column.
    """
    logger.info("Computing residuals.")
    return df.with_columns(
        (pl.col(true_column) - pl.col(pred_column)).alias(residual_name)
    )


def get_residual_sum_square(
    df: pl.DataFrame, residual_column: str = "residual"
) -> float:
    """
    Calculate the residual sum of squares.

    Args:
        df: The input DataFrame.
        residual_column: The name of the residuals column.

    Returns:
        The residual sum of squares.
    """
    residuals = df.get_column(residual_column).to_numpy()
    return float(np.sum(residuals**2))


def calculate_residual_bounds(rss: float, n_rows: int) -> tuple[float, float]:
    """
    Calculate upper and lower residual bounds.

    Args:
        rss: The residual sum of squares.
        n_rows: The number of rows in the DataFrame.

    Returns:
        A tuple containing the lower and upper residual bounds.
    """
    rss_normed = rss / n_rows
    rss_upper = 3 * rss_normed ** (1 / 2)
    rss_lower = -rss_upper
    return rss_lower, rss_upper


def get_residual_bounds(
    df: pl.DataFrame,
    residual_column: str = "residual",
) -> tuple[float, float]:
    """
    Calculate residual bounds for the DataFrame.

    Args:
        df: The input DataFrame.
        residual_column: The name of the residuals column.

    Returns:
        A tuple containing the lower and upper residual bounds.
    """
    n_rows = df.shape[0]
    rss = get_residual_sum_square(df, residual_column=residual_column)
    return calculate_residual_bounds(rss, n_rows)
