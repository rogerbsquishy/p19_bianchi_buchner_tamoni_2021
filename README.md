p19_bianchi_buchner_tamoni_2021
===============================

## About this project

Bond Risk Premiums with Machine Learning

This project replicates the out-of-sample bond excess return forecasts of
Bianchi, Büchner, and Tamoni (2021, *Review of Financial Studies*), "Bond Risk
Premiums with Machine Learning". The authors' code is in
`ProvidedReplicationCode/`. The replication code is in `src/`.

## Replication pipeline

| Step | Script | Output |
|------|--------|--------|
| GSW zero-coupon yields (Fed) | `src/load_fed_yield_curve.py` | `_data/fed_yield_curve.parquet` |
| FRED-MD macro panel | `src/pull_fred_md.py` | `_data/fred_md_raw.parquet`, `_data/fred_md_tcodes.parquet` |
| Forward rates, excess returns, transformed macro data | `src/build_bbt_dataset.py` | `_data/bbt_*.parquet` |
| Recursive out-of-sample forecasts | `src/forecast_oos.py` (models in `src/bbt_models.py`, `src/bbt_nn.py`) | `_data/oos_forecasts/<model>.parquet` |
| R²OOS table, significance, figure | `src/evaluate_oos.py` | `_output/bbt_r2oos*.csv`, `_output/bbt_r2oos_table.tex`, `_output/bbt_cumulative_sse.png` |

Run everything with `doit` (or a single step, e.g. `doit forecast_oos:ElasticNet_macro_fwd`).

**Data.** Predictors are the 1- to 10-year one-year forward rates from the
Gürkaynak–Sack–Wright yield curve and the FRED-MD macro panel, transformed
to be stationary with the FRED-MD transformation codes. Targets are 12-month
excess log returns on 2-, 3-, 4-, 5-, 7-, and 10-year zero-coupon bonds.
Forecast origins run from 1971:08 to 2017:12, so returns are realized through
2018:12. Out-of-sample forecasts start at the 1989:01 origin.

**FRED-MD source.** The St. Louis Fed file server rejects scripted downloads.
For the official data, download a vintage CSV from the
[FRED-MD page](https://www.stlouisfed.org/research/economists/mccracken/fred-databases)
and save it as `data_manual/fred_md.csv`. Without that file,
`pull_fred_md.py` rebuilds the panel series by series from FRED. That panel
is missing the series FRED does not host with full history, including the
stock-market group.

**Models.** The models are the historical-mean benchmark, PCA regression,
PLS, ridge, lasso, elastic net, random forest, and extremely randomized
trees, each run on forward rates only and on macro data plus forward rates.
There are also three neural networks ported from Keras to PyTorch:
`NN1Layer_fwd`, `NN3LayerExog` (32-16-8), and `NN1LayerEnsemExog`, the
group ensemble.

**Timing.** By default, each training window ends 12 months before the
forecast origin, so every training target has been realized by the forecast
date. The provided code trains on rows up to `t-1`. Its last 11 targets are
returns that are not yet known at `t`. To reproduce that behavior, set
`BBT_NO_LOOKAHEAD=False`.

See `.env.example` for the sample dates, `nMC`/`nAvg`, and other settings.

## Quick Start

The quickest way to run code in this repo is to use the following steps.

You must have TexLive (or another LaTeX distribution) installed on your computer and available in your path.
You can do this by downloading and installing it from here ([windows](https://tug.org/texlive/windows.html#install)
and [mac](https://tug.org/mactex/mactex-download.html) installers).


First, you must have the `conda` package manager installed (e.g., via Anaconda). However, I recommend using `mamba`, via [miniforge](https://github.com/conda-forge/miniforge) as it is faster and more lightweight than `conda`.

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate p19_bianchi_buchner_tamoni_2021
```

Finally, run the project tasks:
```bash
doit
```
And that's it!


### Other commands

#### Unit Tests and Doc Tests

You can run the unit test, including doctests, with the following command:
```
pytest --doctest-modules
```

You can build the documentation with:
```
rm ./src/.pytest_cache/README.md
jupyter-book build -W ./
```
Use `del` instead of rm on Windows


#### Setting Environment Variables

You can [export your environment variables](https://stackoverflow.com/questions/43267413/how-to-set-environment-variables-from-env-file)
from your `.env` files like so, if you wish. This can be done easily in a Linux or Mac terminal with the following command:
```bash
set -a  # automatically export all variables
source .env
set +a
```
On Windows (PowerShell):
```powershell
Get-Content .env | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
```

### Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting Python code.

```bash
# Auto-fix linting issues (e.g., unused imports, undefined names)
ruff check . --fix

# Format code (consistent style, spacing, line length)
ruff format .

# Sort imports, then fix linting issues, then format
ruff format . && ruff check --select I --fix . && ruff check --fix .
```

- `ruff check --fix` applies safe auto-fixes for linting violations
- `ruff format` formats code similar to Black
- `--select I` targets only import sorting rules (isort-compatible)

### General Directory Structure

 - The `assets` folder is used for things like hand-drawn figures or other
   pictures that were not generated from code. These things cannot be easily
   recreated if they are deleted.

 - The `_output` folder, on the other hand, contains dataframes and figures that are
   generated from code. The entire folder should be able to be deleted, because
   the code can be run again, which would again generate all of the contents.

 - The `data_manual` is for data that cannot be easily recreated. This data
   should be version controlled. Anything in the `_data` folder or in
   the `_output` folder should be able to be recreated by running the code
   and can safely be deleted.

 - I'm using the `doit` Python module as a task runner. It works like `make` and
   the associated `Makefile`s. To rerun the code, install `doit`
   (https://pydoit.org/) and execute the command `doit` from the `src`
   directory. Note that doit is very flexible and can be used to run code
   commands from the command prompt, thus making it suitable for projects that
   use scripts written in multiple different programming languages.

 - I'm using the `.env` file as a container for absolute paths that are private
   to each collaborator in the project. You can also use it for private
   credentials, if needed. It should not be tracked in Git.

### Data and Output Storage

I'll often use a separate folder for storing data. Any data in the data folder
can be deleted and recreated by rerunning the PyDoit command (the pulls are in
the dodo.py file). Any data that cannot be automatically recreated should be
stored in the "data_manual" folder. Because of the risk of manually-created data
getting changed or lost, I prefer to keep it under version control if I can.
Thus, data in the "_data" folder is excluded from Git (see the .gitignore file),
while the "data_manual" folder is tracked by Git.

Output is stored in the "_output" directory. This includes dataframes, charts, and
rendered notebooks. When the output is small enough, I'll keep this under
version control. I like this because I can keep track of how dataframes change as my
analysis progresses, for example.

Of course, the _data directory and _output directory can be kept elsewhere on the
machine. To make this easy, I always include the ability to customize these
locations by defining the path to these directories in environment variables,
which I intend to be defined in the `.env` file, though they can also simply be
defined on the command line or elsewhere. The `settings.py` is responsible for
loading these environment variables and doing some preprocessing on them.
The `settings.py` file is the entry point for all other scripts to these
definitions. That is, all code that references these variables and others are
loaded by importing `config`.

### Naming Conventions

 - **`pull_` vs `load_`**: Files or functions that pull data from an external
 data source are prepended with "pull_", as in "pull_fred.py". Functions that
 load data that has been cached in the "_data" folder are prepended with "load_".
 For example, inside of the `pull_CRSP_Compustat.py` file there is both a
 `pull_compustat` function and a `load_compustat` function. The first pulls from
 the web, whereas the other loads cached data from the "_data" directory.


### Dependencies and Virtual Environments

#### Working with `conda` environments

This project uses conda for environment management. The dependencies are stored in `environment.yml`.

To create/update the environment:
```bash
conda env create -f environment.yml
# or to update an existing environment:
conda env update -f environment.yml
```

To activate the environment:
```bash
conda activate p19_bianchi_buchner_tamoni_2021
```

To export the current environment:
```bash
conda env export > environment.yml
```

**Tip:** Consider using `mamba` instead of `conda` for faster package resolution. Install via [miniforge](https://github.com/conda-forge/miniforge).

