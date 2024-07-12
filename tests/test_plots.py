from vaxstats.io import load_file
from vaxstats.plots import plot_data_line, plot_forecast
from vaxstats.utils import datetime_to_float


def test_data_plot(path_example_prepped_csv, path_example_img):
    df = load_file(path_example_prepped_csv, file_type="csv")
    temps = df["y"].to_numpy()
    days = datetime_to_float(df, time_unit="days")
    fig = plot_data_line(days, temps)
    fig.savefig(path_example_img)  # type: ignore


def test_data_forecast_plot(path_example_forecast_csv, path_example_img):
    df = load_file(path_example_forecast_csv, file_type="csv")
    temps = df["y"].to_numpy()
    temps_pred = df["y_hat"].to_numpy()
    days = datetime_to_float(df, time_unit="days")
    fig = plot_forecast(days, temps, temps_pred)
    fig.savefig(path_example_img)  # type: ignore
