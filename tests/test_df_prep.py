import os

from vaxstats.cli import main
from vaxstats.io import clean_df, load_file, prep_forecast_df


def test_load_excel(path_example_excel):
    df = load_file(path_example_excel)
    assert df.shape == (2_742, 10)


def test_clean_df(path_example_excel):
    df = load_file(path_example_excel)
    df = clean_df(df)
    assert df.shape == (2_742, 10)


def test_prep_df(path_example_excel):
    df = load_file(path_example_excel)
    df = clean_df(df)
    df = prep_forecast_df(df, 0, 1, 6)
    assert df.columns == ["unique_id", "ds", "y"]
    assert df.shape == (2_721, 3)


def test_prep_cli(capsys, monkeypatch):
    output_path = "tests/tmp/data-prepped.csv"
    if os.path.exists(output_path):
        os.remove(output_path)
    cmd = [
        "vaxstats",
        "prep",
        "tests/files/example.xlsx",
        "--date_idx",
        "0",
        "--time_idx",
        "1",
        "--y_idx",
        "6",
        "--output",
        output_path,
    ]
    monkeypatch.setattr("sys.argv", cmd)
    main()
    captured = capsys.readouterr()
    print(captured)
    assert "usage: vaxstats" not in captured
    assert os.path.exists(output_path)
