# `peak`

**Invoked by:** `vaxstats peak`

The `peak` command allows you to quickly preview how polars will load your data file.
This is particularly useful for verifying that your file is being read correctly before proceeding with further data processing or analysis.

## Usage

```bash
vaxstats peak <file_path> [-n <number_of_rows>]
```

## Examples

```bash
vaxstats peak data.csv
```

```bash
vaxstats peak experiment_results.xlsx -n 5
```

## Arguments

| Argument | Description | Required | Default |
| -------- | ----------- | -------- | ------- |
| `file_path` | Path to the file you want to peek into. Can be CSV, Excel, or other formats supported by polars. | Yes | N/A |

## Options

| Option | Description | Default | Example |
| ------ | ----------- | ------- | ------- |
| `-n` | Number of rows to display in the preview. | 3 | `-n 10` |

## Output

The command will print the first `n` rows of the dataframe as it is loaded by polars.
This output includes:

1.  Column names
2.  Data types inferred by polars for each column
3.  The first `n` rows of data

## Use Cases

1.  **Quick Data Inspection**: Rapidly check the structure and content of your data files.
2.  **Format Verification**: Ensure that polars is interpreting your date, time, and numeric columns correctly.
3.  **Column Identification**: Easily identify the indices of important columns for use with other commands like `prep`.

## Tips

-   If your file has many columns, consider using a larger terminal window or redirecting the output to a file for easier viewing.
-   Pay attention to the data types inferred by polars.
    If they don't match your expectations (e.g., dates loaded as strings), you may need to specify explicit data types when using other commands.
-   Use this command before running `prep` to help you correctly specify column indices and date/time formats.

!!! note
    The `peak` command does not modify your original file; it only reads and displays a preview of the data.

!!! tip
    If you're working with a large file and want to see more context, increase the number of rows displayed using the `-n` option.
    For example: `vaxstats peak large_dataset.csv -n 20`
