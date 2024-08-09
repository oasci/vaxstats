# `forecast`

**Invoked by:** `vaxstats forecast`

The `forecast` command performs forecasting on prepared data using a specified statistical forecasting model. It allows you to apply various forecasting algorithms to your time series data and generate predictions.

## Usage

```bash
vaxstats forecast <file_path> <sf_model> [options]
```

## Examples

```bash
vaxstats forecast data.csv statsforecast.models.ARIMA --baseline_hours 48
```

```bash
vaxstats forecast prepared_data.csv statsforecast.models.ARIMA \
  --baseline_days 3 --sf_model_kwargs "{'order': (1,1,1)}" \
  --output_path forecast_results.csv
```

## Required Arguments

| Argument | Description | Example |
| -------- | ----------- | ------- |
| `file_path` | Path to the input data file (CSV format). | `data.csv`, `prepared_data.csv` |
| `sf_model` | The forecasting model class to be used. | `statsforecast.models.ARIMA` |

## Optional Arguments

| Option | Description | Default | Example |
| ------ | ----------- | ------- | ------- |
| `--baseline_days` | The time window in days for the training set. | N/A | `3`, `7.5` |
| `--baseline_hours` | The time window in hours for the training set. | `72.0` | `48`, `120` |
| `--sf_model_args` | Positional arguments for the forecasting model constructor. | `()` | `"(1, 'string', [1, 2, 3])"` |
| `--sf_model_kwargs` | Keyword arguments for the forecasting model constructor. | `{}` | `"{'order': (1,1,1), 'seasonal_order': (0,1,1,12)}"` |
| `--output_path` | Path to save the output DataFrame with forecasted values. | `output.csv` | `forecast_results.csv` |

!!! note
    You must specify either `--baseline_days` or `--baseline_hours`, but not both.

## Input

The input file should be a CSV file containing at least the following columns:

-   `ds`: Datetime column
-   `y`: Target variable column, $y$

This file is typically the output of the `prep` command or a similarly structured dataset.

## Output

The `forecast` command will generate a CSV file (default name: `output.csv`) with the following columns:

1.  All original columns from the input file
2.  `y_hat`: The forecasted values, $\hat{y}$
3.  `residuals`: The difference between actual and forecasted values ($y - \hat{y}$)

## Forecasting Model Specification
-   Use the `sf_model` argument to specify the full path to the forecasting model class.
-   Use `--sf_model_args` and `--sf_model_kwargs` to pass additional arguments to the model constructor.
-   Arguments should be specified using Python literal syntax.

## Examples of Model Specifications

1.  ARIMA model with specific parameters:
    ```bash
    vaxstats forecast data.csv statsforecast.models.ARIMA \
        --sf_model_kwargs "{'order': (1,1,1), 'seasonal_order': (0,1,1,12)}"
    ```
2.  Simple exponential smoothing:
    ```bash
    vaxstats forecast data.csv statsforecast.models.SimpleExponentialSmoothing
    ```

!!! tip
    Experiment with different baseline windows and model parameters to find the best fit for your data. You can compare the residuals to evaluate model performance.
