"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
from os import environ
from pathlib import Path

from settings import config

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
USER = config("USER")

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["python ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_pull():
    """Pull data from external sources"""
    yield {
        "name": "fred",
        "doc": "Pull data from FRED",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_fred.py",
        ],
        "targets": [DATA_DIR / "fred.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_fred.py"],
        "clean": [],
    }
    yield {
        "name": "ofr",
        "doc": "Pull data from OFR API",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_ofr_api_data.py",
        ],
        "targets": [DATA_DIR / "ofr_public_repo_data.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_ofr_api_data.py"],
        "clean": [],
    }
    yield {
        "name": "crsp_stock",
        "doc": "Pull CRSP stock data from WRDS",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_CRSP_stock.py",
        ],
        "targets": [DATA_DIR / "CRSP_monthly_stock.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_CRSP_stock.py"],
        "clean": [],
    }
    yield {
        "name": "crsp_compustat",
        "doc": "Pull CRSP Compustat data from WRDS",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_CRSP_Compustat.py",
        ],
        "targets": [DATA_DIR / "CRSP_Compustat.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_CRSP_Compustat.py"],
        "clean": [],
    }


##################################################################
## Replication of Bianchi, Buchner, and Tamoni (2021)
##################################################################

BBT_SKLEARN_MODELS = [
    "HistoricalMean",
    "PCA_fwd", "PLS_fwd", "Ridge_fwd", "Lasso_fwd", "ElasticNet_fwd", "RF_fwd", "ET_fwd",
    "PCA_macro_fwd", "PLS_macro_fwd", "Ridge_macro_fwd", "Lasso_macro_fwd",
    "ElasticNet_macro_fwd", "RF_macro_fwd", "ET_macro_fwd",
]  # fmt: skip
BBT_NN_MODELS = ["NN1Layer_fwd", "NN3LayerExog", "NN1LayerEnsemExog"]
BBT_DATASET = [
    DATA_DIR / f
    for f in [
        "bbt_forward_rates.parquet",
        "bbt_macro.parquet",
        "bbt_excess_returns.parquet",
        "bbt_macro_groups.parquet",
    ]
]


def task_pull_bbt():
    """Pull GSW zero-coupon yields and the FRED-MD macro panel"""
    yield {
        "name": "gsw_yields",
        "doc": "Gurkaynak, Sack, and Wright (2007) yield curve from the Fed",
        "actions": ["python ./src/settings.py", "python ./src/load_fed_yield_curve.py"],
        "targets": [DATA_DIR / "fed_yield_curve.parquet"],
        "file_dep": ["./src/settings.py", "./src/load_fed_yield_curve.py"],
        "clean": [],
    }
    yield {
        "name": "fred_md",
        "doc": "FRED-MD (manual file, official URL, or rebuilt from FRED)",
        "actions": ["python ./src/settings.py", "python ./src/pull_fred_md.py"],
        "targets": [
            DATA_DIR / "fred_md_raw.parquet",
            DATA_DIR / "fred_md_tcodes.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_fred_md.py",
            "./src/fred_md_metadata.py",
        ],
        "clean": [],
    }


def task_build_bbt_dataset():
    """Forward rates, excess returns, and transformed macro predictors"""
    return {
        "actions": ["python ./src/build_bbt_dataset.py"],
        "targets": BBT_DATASET,
        "file_dep": [
            "./src/build_bbt_dataset.py",
            DATA_DIR / "fed_yield_curve.parquet",
            DATA_DIR / "fred_md_raw.parquet",
            DATA_DIR / "fred_md_tcodes.parquet",
        ],
        "clean": True,
    }


def task_forecast_oos():
    """Recursive out-of-sample forecasts, one sub-task per model"""
    for model in BBT_SKLEARN_MODELS + BBT_NN_MODELS:
        model_file = "bbt_nn.py" if model in BBT_NN_MODELS else "bbt_models.py"
        yield {
            "name": model,
            "actions": [f"python ./src/forecast_oos.py --models {model}"],
            "targets": [DATA_DIR / "oos_forecasts" / f"{model}.parquet"],
            "file_dep": [
                "./src/forecast_oos.py",
                f"./src/{model_file}",
                *BBT_DATASET,
            ],
            "clean": True,
            "verbosity": 2,
        }


def task_evaluate_oos():
    """R2OOS table and figure"""
    return {
        "actions": ["python ./src/evaluate_oos.py"],
        "targets": [
            OUTPUT_DIR / "bbt_r2oos.csv",
            OUTPUT_DIR / "bbt_r2oos_pvalues.csv",
            OUTPUT_DIR / "bbt_r2oos_table.tex",
            OUTPUT_DIR / "bbt_cumulative_sse.png",
        ],
        "file_dep": [
            "./src/evaluate_oos.py",
            *[
                DATA_DIR / "oos_forecasts" / f"{m}.parquet"
                for m in BBT_SKLEARN_MODELS + BBT_NN_MODELS
            ],
        ],
        "clean": True,
        "verbosity": 2,
    }


def task_summary_stats():
    """Generate summary statistics tables"""
    file_dep = ["./src/example_table.py"]
    file_output = [
        "example_table.tex",
        "pandas_to_latex_simple_table1.tex",
    ]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            "python ./src/example_table.py",
            "python ./src/pandas_to_latex_demo.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_example_plot():
    """Example plots"""
    file_dep = [Path("./src") / file for file in ["example_plot.py", "pull_fred.py"]]
    file_output = ["example_plot.png"]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            "python ./src/example_plot.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_chart_repo_rates():
    """Example charts for Chartbook"""
    file_dep = [
        "./src/pull_fred.py",
        "./src/chart_relative_repo_rates.py",
    ]
    targets = [
        DATA_DIR / "repo_public.parquet",
        DATA_DIR / "repo_public.xlsx",
        DATA_DIR / "repo_public_relative_fed.parquet",
        DATA_DIR / "repo_public_relative_fed.xlsx",
        OUTPUT_DIR / "repo_rates.html",
        OUTPUT_DIR / "repo_rates_normalized.html",
        OUTPUT_DIR / "repo_rates_normalized_w_balance_sheet.html",
    ]

    return {
        "actions": [
            "python ./src/chart_relative_repo_rates.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


notebook_tasks = {
    "01_example_notebook_interactive.ipynb.py": {
        "path": "./src/01_example_notebook_interactive.ipynb.py",
        "file_dep": [],
        "targets": [],
    },
    "02_example_with_dependencies.ipynb.py": {
        "path": "./src/02_example_with_dependencies.ipynb.py",
        "file_dep": ["./src/pull_fred.py"],
        "targets": [OUTPUT_DIR / "GDP_graph.png"],
    },
    "03_public_repo_summary_charts.ipynb.py": {
        "path": "./src/03_public_repo_summary_charts.ipynb.py",
        "file_dep": [
            "./src/pull_fred.py",
            "./src/pull_ofr_api_data.py",
            "./src/pull_public_repo_data.py",
        ],
        "targets": [
            OUTPUT_DIR / "repo_rate_spikes_and_relative_reserves_levels.png",
            OUTPUT_DIR / "rates_relative_to_midpoint.png",
        ],
    },
}


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix("")  # strips .py, leaves .ipynb
        notebook_name = notebook_path.stem  # e.g. "01_example_notebook_interactive"
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on

###############################################################
## Task below is for LaTeX compilation
###############################################################


def task_compile_latex_docs():
    """Compile the LaTeX documents to PDFs"""
    file_dep = [
        "./reports/report_example.tex",
        "./reports/my_article_header.sty",
        "./reports/slides_example.tex",
        "./reports/my_beamer_header.sty",
        "./reports/my_common_header.sty",
        "./reports/report_simple_example.tex",
        "./reports/slides_simple_example.tex",
        # Generated inputs. The reports typeset a placeholder for a missing one
        # instead of failing, so list them here to have doit build them first.
        OUTPUT_DIR / "example_plot.png",
        OUTPUT_DIR / "example_table.tex",
        OUTPUT_DIR / "pandas_to_latex_simple_table1.tex",
    ]
    targets = [
        "./reports/report_example.pdf",
        "./reports/slides_example.pdf",
        "./reports/report_simple_example.pdf",
        "./reports/slides_simple_example.pdf",
    ]

    return {
        "actions": [
            # My custom LaTeX templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_example.tex",  # Clean
            # Simple templates based on small adjustments to Overleaf templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_simple_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_simple_example.tex",  # Clean
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }

sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Compile Sphinx Docs"""
    notebook_scripts = [
        Path(notebook_tasks[notebook]["path"])
        for notebook in notebook_tasks.keys()
    ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        *notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook docs build -f",
        ],  # Use docs as build destination
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": [
            "run_notebooks",
        ],
        "clean": True,
    }


def task_run_pytest():
    """Run pytest and save results to OUTPUT_DIR"""
    src_py_files = list(Path("./src").glob("*.py"))
    test_output = OUTPUT_DIR / "pytest_results.xml"

    def run_pytest():
        import subprocess

        result = subprocess.run(
            ["pytest", f"--junitxml={test_output}"],
        )
        if result.returncode != 0:
            # Remove the XML so doit won't consider the target up-to-date
            Path(test_output).unlink(missing_ok=True)
            raise RuntimeError(f"pytest failed with exit code {result.returncode}")

    return {
        "actions": [run_pytest],
        "targets": [test_output],
        "file_dep": src_py_files,
        "clean": True,
        "verbosity": 2,
    }
