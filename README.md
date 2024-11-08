<h1 align="center">vaxstats</h1>

<h4 align="center">Help with statistical forecasting models for vaccine studies.</h4>

<p align="center">
    <a href="https://github.com/oasci/vaxstats/actions/workflows/tests.yml">
        <img src="https://github.com/oasci/vaxstats/actions/workflows/tests.yml/badge.svg" alt="Build Status ">
    </a>
    <!-- <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/vaxstats"> -->
    <a href="https://codecov.io/gh/oasci/vaxstats">
        <img src="https://codecov.io/gh/oasci/vaxstats/branch/main/graph/badge.svg" alt="codecov">
    </a>
    <!-- <a href="https://github.com/oasci/vaxstats/releases">
        <img src="https://img.shields.io/github/v/release/oasci/vaxstats" alt="GitHub release (latest by date)">
    </a> -->
    <a href="https://github.com/oasci/vaxstats/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/github/license/oasci/vaxstats" alt="License">
    </a>
    <a href="https://github.com/oasci/vaxstats/" target="_blank">
        <img src="https://img.shields.io/github/repo-size/oasci/vaxstats" alt="GitHub repo size">
    </a>
</p>

<h4 align="center" style="padding-bottom: 0.5em;"><a href="https://durrantlab.github.io/wisp/">Documentation</a></h4>

`vaxstats` is a powerful tool designed to support statistical forecasting models tailored specifically for vaccine studies.
It provides a robust command-line interface and modular architecture, enabling researchers to efficiently analyze, forecast, and visualize vaccine-related data.
This tool is particularly useful in predicting trends and understanding the impact of vaccines over time.

## Features

-   **Data Preparation:** Clean and prepare vaccine-related datasets for analysis.
-   **Forecasting:** Apply statistical forecasting models to predict future trends based on historical data.
-   **Visualization:** Generate insightful visualizations to aid in the interpretation of forecasting results.
-   **Modular Design:** Easily extendable with custom models and analysis routines.

## Acknowledgements

This project was developed as part of [OASCI's](https://www.oasci.org/) [Scientific Computing Core (SC2)](https://thescientific.cc/) initiative to enhance interdisciplinary computational research.
We appreciate contributions and feedback from the broader community.

In particular, we designed and implemented this software in collaboration with the following group(s):

-   [Dr. Doug Reed](https://www.aerobiology-at-pitt.com/) in the [Center for Vaccine Research](https://www.cvr.pitt.edu/) at the [University of Pittsburgh](https://www.pitt.edu/).

## Installation

Clone the [repository](https://github.com/oasci/vaxstats):

```bash
git clone https://github.com/oasci/vaxstats.git
```

Install `vaxstats` using `pip` after moving into the directory.

```sh
pip install .
```

This will install all dependencies and `vaxstats` into your current Python environment.

## Development

We use [pixi](https://pixi.sh/latest/) to manage Python environments and simplify the developer workflow.
Once you have [pixi](https://pixi.sh/latest/) installed, move into `vaxstats` directory (e.g., `cd vaxstats`) and install the  environment using the command

```bash
pixi install
```

Now you can activate the new virtual environment using

```sh
pixi shell
```

## License

This project is released under the Apache-2.0 License as specified in `LICENSE.md`.
