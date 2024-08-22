# `analyze`

**Invoked by:** `vaxstats analyze`

The `analyze` command performs statistical analysis on forecasted data, calculating various metrics and thresholds for temperature anomalies.

## Usage

```bash
vaxstats analyze <file_path> [options]
```

## Examples

```bash
vaxstats analyze forecasted_data.csv --baseline_hours 48
```

```bash
vaxstats analyze results.csv --baseline_days 3 \
--data_column "temperature" --pred_column "forecast" \
--output_path analysis_results.json
```

## Required Arguments

| Argument | Description | Example |
| -------- | ----------- | ------- |
| `file_path` | Path to the forecasted output file (CSV format). | `forecasted_data.csv`, `results.csv` |

## Optional Arguments

| Option | Description | Default | Example |
| ------ | ----------- | ------- | ------- |
| `--baseline_days` | The time window in days for the baseline calculation. | N/A | `3`, `7.5` |
| `--baseline_hours` | The time window in hours for the baseline calculation. | N/A | `48`, `120` |
| `--data_column` | Column name containing observed data. | `y` | `temperature` |
| `--pred_column` | Column name containing predicted data. | `y_hat` | `forecast` |
| `--residual_column` | Column name containing residuals. | `residual` | `error` |
| `--date_column` | Column name containing datetime data. | `ds` | `timestamp` |
| `--datetime_fmt` | Format of the input date strings. | `%Y-%m-%d %H:%M:%S` | `%m/%d/%Y %H:%M` |
| `--output_path` | Path to save the output JSON with analysis results. | `analysis.json` | `results.json` |

!!! note
    You must specify either `--baseline_days` or `--baseline_hours`, but not both.

## Input

The input file should be a CSV file containing at least the following columns:

-   Datetime column (specified by `--date_column`)
-   Observed data column (specified by `--data_column`)
-   Predicted data column (specified by `--pred_column`)
-   Residuals column (specified by `--residual_column`)

This file is typically the output of the `forecast` command or a similarly structured dataset.

## Output

The `analyze` command will generate a JSON file (default name: `analysis.json`) with the following structure:

```json
{
  "baseline_stats": {
    "degrees_of_freedom": int,
    "average_temp": float,
    "std_dev_temp": float,
    "max_temp": float,
    "min_temp": float,
    "residual_sum_squares": float
  },
  "residual_stats": {
    "max_residual": float,
    "residual_lower_bound": float,
    "residual_upper_bound": float
  },
  "duration_stats": {
    "total_duration_hours": float,
    "fever_hours": int,
    "hypothermia_hours": int
  }
}
```

## Analysis Process

1.  Loads the input file and converts the date column to datetime format.
2.  Calculates baseline statistics using the specified time window.
3.  Computes hourly statistics, including median observed and predicted values.
4.  Determines fever and hypothermia thresholds based on residual bounds.
5.  Calculates various statistics including temperature ranges, residual information, and duration of anomalies.

## Hourly Statistics

The analysis includes hourly statistics with the following information:

-   Median observed value
-   Median predicted value
-   Start and end time of the hour
-   Number of data points in the hour
-   Hypothermia threshold
-   Fever threshold
