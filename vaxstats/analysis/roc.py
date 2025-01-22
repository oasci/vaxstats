import numpy as np
import polars as pl
from loguru import logger


def get_sum_sqr_success_diffs(df: pl.DataFrame, column_name: str) -> float:
    """
    Calculate the Sum of Squared Successive Differences (SSSD) based on
    consecutive temperature measurements.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to compute SSSD.

    Returns:
        The Sum of Squared Successive Differences (SSSD).
    """
    # Extract the baseline temperature data
    baseline = df[column_name].to_numpy()

    # Compute the differences between consecutive temperature measurements
    diffs = np.diff(baseline)

    # Identify valid differences where neither of the consecutive measurements is NaN
    valid_mask = ~np.isnan(diffs) & ~np.isnan(baseline[:-1]) & ~np.isnan(baseline[1:])

    # Calculate squared differences, setting invalid entries to zero
    squared_diffs = np.where(valid_mask, diffs**2, 0)

    # Sum the squared differences to obtain SSSD
    sum_squared_successive_diffs = squared_diffs.sum()

    logger.debug(f"SSSD calculated: {sum_squared_successive_diffs}")
    return float(sum_squared_successive_diffs)


def get_roc_bounds(
    df: pl.DataFrame,
    column_name: str,
) -> tuple[float, float]:
    """
    Calculate rate of change bounds for the DataFrame.

    Args:
        df: The input DataFrame.
        column_name: The name of the column.

    Returns:
        Lower ROC bound for hypothermia detection.

        Upper ROC bound for fever detection.
    """
    n_rows = df.shape[0]
    sssd = get_sum_sqr_success_diffs(df, column_name=column_name)
    mean_sssd = sssd / (n_rows - 1)
    bound_upper = float(3 * np.sqrt(mean_sssd))
    bound_lower = -bound_upper
    return bound_lower, bound_upper
