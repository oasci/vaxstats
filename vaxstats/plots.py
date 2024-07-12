from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_data_line(
    time: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    xlabel: str = "Time",
    ylabel: str = "Temperature [°C]",
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True, **kwargs)  # type: ignore
    ax.plot(time, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig  # type: ignore
