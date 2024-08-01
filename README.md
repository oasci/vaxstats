<h1 align="center">vaxstats</h1>

<h4 align="center">Help with statistical forecasting models for vaccine studies.</h4>

Add information about `vaxstats` here.

## Installation

Clone the [repository](https://github.com/oasci/vaxstats):

```bash
git clone https://github.com/oasci/vaxstats.git
```

### Conda environment

Move into `vaxstats` directory (`cd vaxstats`) and install the development conda environment using [GNU Make](https://www.gnu.org/software/make/) (which could be installed by default on your system).

```bash
make environment
```

Now you can activate the new conda environment `vaxstats-dev` and use `vaxstats` commands.

```sh
conda activate vaxstats-dev
```

### Manual install

Alternatively, you can manually install `vaxstats` using `pip` after moving into the directory.

```sh
pip install .
```

This will install all dependencies and `vaxstats` into your current Python environment.

## Deploying

We use [bump-my-version](https://github.com/callowayproject/bump-my-version) to release a new version.
This will create a git tag used by [poetry-dynamic-version](https://github.com/mtkennerly/poetry-dynamic-versioning) to generate version strings and update `CHANGELOG.md`.

For example, you would run the following command to bump the `minor` version.

```bash
poetry run bump-my-version bump minor
```

After releasing a new version, you must push and include all tags.

```bash
git push --follow-tags
```

## License

This project is released under the Apache-2.0 License as specified in `LICENSE.md`.
