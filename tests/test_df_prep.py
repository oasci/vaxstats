import os
from datetime import datetime

from vaxstats.cli import main
from vaxstats.io import clean_df, load_file, prep_forecast_df


def test_load_excel(path_example_excel):
    df = load_file(path_example_excel)
    assert df.shape == (2_742, 10)


def test_load_excel_m9324(path_m9324_excel):
    df = load_file(path_m9324_excel, sheet_name="Step 2")
    assert df.shape == (1_762, 10)


def test_clean_df(path_example_excel):
    df = load_file(path_example_excel)
    df = clean_df(df)
    assert df.shape == (2_742, 10)


def test_clean_excel_m9324(path_m9324_excel):
    df = load_file(path_m9324_excel, sheet_name="Step 2")
    df = clean_df(df)
    assert df.shape == (1_762, 10)


def test_prep_df(path_example_excel):
    df = load_file(path_example_excel)
    df = clean_df(df)
    df = prep_forecast_df(df, 0, 0, 6)
    assert df.columns == ["unique_id", "ds", "y"]
    assert df.shape == (2_721, 3)


def test_prep_excel_m9324(path_m9324_excel):
    df = load_file(path_m9324_excel, sheet_name="Step 2")
    df = clean_df(df)
    df = prep_forecast_df(df, date_idx=0, time_idx=0, y_idx=3)
    assert df.columns == ["unique_id", "ds", "y"]
    assert df["ds"][0] == datetime(2024, 6, 20, 23, 57, 20)
    assert df["y"][0] == 37.049
    assert df["ds"][-1] == datetime(2024, 7, 1, 8, 42, 20)
    assert df["y"][-1] == 35.449
    assert df.shape == (996, 3)


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
