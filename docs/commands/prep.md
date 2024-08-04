# `prep`

**Invoked by:** `vaxstats prep`

The `prep` command is designed to clean and prepare data from various experimental sources for uniform processing in `vaxstats`. It addresses the common challenge of inconsistent data formats by allowing you to specify the crucial data elements and their formats.

## Usage

```bash
vaxstats prep <file_path> --date_idx <index> --time_idx <index> --y_idx <index> [options]
```

## Examples

```bash
vaxstats prep example.xlsx --date_idx 0 --time_idx 1 --y_idx 6
```

```bash
vaxstats prep ./data/2024-2-8.xlsx --date_idx 3 --time_idx 2 --y_idx 4 --input_date_fmt "%Y-%m-%d"
```

## Required Arguments

| Argument | Description | Example |
| -------- | ----------- | ------- |
| `file_path` | Path to Excel or CSV file containing the data. | `2024-08-04.xlsx`, `data/exp001.csv` |
| `--date_idx` | Column index containing **date** information. | `0`, `1`, etc. |
| `--time_idx` | Column index containing **time** information. | `0`, `1`, etc. |
| `--y_idx` | Column index containing the **target variable** (e.g., temperatures). | `0`, `1`, etc. |

!!! note
    All column indices are zero-based, meaning the first column has index `0`.

## Optional Arguments

| Option | Description | Default | Example |
| ------ | ----------- | ------- | ------- |
| `--input_date_fmt` | Format of the input date strings. | `%m-%d-%y` | `%Y-%m-%d`, `%d/%m/%Y` |
| `--input_time_fmt` | Format of the input time strings. | `%I:%M:%S %p` | `%H:%M:%S`, `%I:%M %p` |
| `--output_fmt` | Format of the output datetime strings. | `%Y-%m-%d %H:%M:%S` | `%Y-%m-%dT%H:%M:%S` |
| `--output` | Name of the output CSV file. | `output.csv` | `prepared_data.csv` |

## Input Format Specifications

-   For `--input_date_fmt` and `--input_time_fmt`, use Python's [strftime format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes).
-   Common format specifiers:
    -   `%Y`: Year with century (e.g., 2024)
    -   `%y`: Year without century as zero-padded decimal number (01, 02, 99).
    -   `%m`: Month as a zero-padded number (01-12)
    -   `%d`: Day of the month as a zero-padded number (01-31)
    -   `%H`: Hour (24-hour clock) as a zero-padded number (00-23)
    -   `%I`: Hour (12-hour clock) as a zero-padded number (01-12)
    -   `%M`: Minute as a zero-padded number (00-59)
    -   `%S`: Second as a zero-padded number (00-59)
    -   `%p`: Locale's equivalent of AM or PM

## Output

The `prep` command will generate a CSV file (default name: `output.csv`) with the following columns:

1.  `unique_id`: A unique identifier for each row (always set to 0 in the current version).
2.  `ds`: The combined and formatted date and time.
3.  `y`: The target variable (from the column specified by `--y_idx`).

This output format is designed to be compatible with various forecasting and analysis tools.

!!! tip
    If your input data has a non-standard date or time format, use the `--input_date_fmt` and `--input_time_fmt` options to specify the correct format. This ensures accurate parsing of your datetime information.
