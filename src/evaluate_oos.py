"""Evaluate out-of-sample forecasts: R2OOS and its significance.

Ports ``R2OOS`` and ``RSZ_Signif`` from the provided replication code. The
benchmark is the historical-mean forecast produced by ``forecast_oos.py``
(model ``HistoricalMean``), so it uses exactly the training window of the
other models.

    R2OOS = 1 - sum (y - y_hat)^2 / sum (y - y_bar)^2

Significance follows Rapach, Strauss, and Zhou (2010) / Clark and West (2007):
regress ``(y - y_bar)^2 - (y - y_hat)^2 + (y_bar - y_hat)^2`` on a constant
with Newey-West (12 lags) standard errors and report the one-sided p-value.

Outputs (in ``OUTPUT_DIR``):

- ``bbt_r2oos.csv`` / ``bbt_r2oos_pvalues.csv``: R2OOS (percent) and p-values,
  models x maturities.
- ``bbt_r2oos_table.tex``: LaTeX table in the layout of the paper's main
  results table (stars for 10%, 5%, 1% significance).
- ``bbt_cumulative_sse.png``: cumulative SSE difference versus the benchmark.

With ``BBT_NO_LOOKAHEAD=False`` the file names get a ``_lookahead`` suffix.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as tdist

from forecast_oos import FORECAST_DIR, RUN_SUFFIX
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
BENCHMARK = "HistoricalMean"

# Display order and labels for the results table
MODEL_LABELS = {
    "PCA_fwd": "PCA",
    "PLS_fwd": "PLS",
    "Ridge_fwd": "Ridge",
    "Lasso_fwd": "Lasso",
    "ElasticNet_fwd": "Elastic net",
    "RF_fwd": "Random forest",
    "ET_fwd": "Extreme trees",
    "NN1Layer_fwd": "NN, 1 layer (3)",
    "PCA_macro_fwd": "PCA",
    "PLS_macro_fwd": "PLS",
    "Ridge_macro_fwd": "Ridge",
    "Lasso_macro_fwd": "Lasso",
    "ElasticNet_macro_fwd": "Elastic net",
    "RF_macro_fwd": "Random forest",
    "ET_macro_fwd": "Extreme trees",
    "NN3LayerExog": "NN, 3 layers (32-16-8)",
    "NN1LayerEnsemExog": "NN, group ensemble",
}
PANELS = {
    "A. Forward rates only": [
        k for k in MODEL_LABELS if k.endswith("_fwd") and "macro" not in k
    ],
    "B. Macro variables and forward rates": [
        k for k in MODEL_LABELS if "macro" in k or k.startswith("NN3") or "Ensem" in k
    ],
}


def r2_oos(y_true, y_forecast, y_bench):
    """Out-of-sample R^2 relative to a benchmark forecast.

    Observations where any input is missing are ignored.

    >>> y = np.array([1.0, 2.0, 3.0])
    >>> round(r2_oos(y, y, np.zeros(3)), 6)
    1.0
    >>> round(r2_oos(y, np.zeros(3), np.zeros(3)), 6)
    0.0
    """
    y_true, y_forecast, y_bench = map(np.asarray, (y_true, y_forecast, y_bench))
    ok = ~(np.isnan(y_true) | np.isnan(y_forecast) | np.isnan(y_bench))
    ss_res = np.sum(np.square(y_true[ok] - y_forecast[ok]))
    ss_tot = np.sum(np.square(y_true[ok] - y_bench[ok]))
    return float(1 - ss_res / ss_tot)


def clark_west_pvalue(y_true, y_forecast, y_bench, maxlags=12):
    """One-sided p-value for H0: R2OOS <= 0 (``RSZ_Signif`` in provided code)."""
    y_true, y_forecast, y_bench = map(np.asarray, (y_true, y_forecast, y_bench))
    ok = ~(np.isnan(y_true) | np.isnan(y_forecast) | np.isnan(y_bench))
    y, f, b = y_true[ok], y_forecast[ok], y_bench[ok]
    adj = np.square(y - b) - np.square(y - f) + np.square(b - f)
    res = sm.OLS(adj, np.ones_like(adj)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return float(1 - tdist.cdf(res.tvalues[0], res.nobs - 1))


def load_forecasts(forecast_dir=FORECAST_DIR):
    forecast_dir = Path(forecast_dir)
    return {p.stem: pd.read_parquet(p) for p in sorted(forecast_dir.glob("*.parquet"))}


def evaluate(forecasts, realized, benchmark=BENCHMARK):
    """R2OOS (percent) and p-values for every model and maturity."""
    bench = forecasts[benchmark]
    oos = bench.dropna(how="all").index
    realized = realized.loc[oos]
    r2, pv = {}, {}
    for name, fc in forecasts.items():
        if name == benchmark:
            continue
        fc = fc.loc[oos]
        r2[name] = {
            c: 100 * r2_oos(realized[c], fc[c], bench.loc[oos, c])
            for c in realized.columns
        }
        pv[name] = {
            c: clark_west_pvalue(realized[c], fc[c], bench.loc[oos, c])
            for c in realized.columns
        }
    return pd.DataFrame(r2).T, pd.DataFrame(pv).T


def _stars(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


def to_latex_table(r2, pv, oos_label):
    """LaTeX tabular with one panel per predictor set."""
    cols = list(r2.columns)
    header = " & ".join(f"$rx^{{({c[2:]})}}$" for c in cols)
    lines = [
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        rf"Model & {header} \\",
        r"\midrule",
    ]
    for panel, names in PANELS.items():
        names = [n for n in names if n in r2.index]
        if not names:
            continue
        lines.append(rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{panel}}}}} \\")
        for n in names:
            cells = " & ".join(
                f"${r2.loc[n, c]:.1f}{_stars(pv.loc[n, c])}$" for c in cols
            )
            lines.append(rf"\quad {MODEL_LABELS[n]} & {cells} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        "",
        rf"% R2OOS in percent relative to the historical mean, {oos_label}.",
        "% Stars: Clark-West one-sided p-value < 0.10 (*), 0.05 (**), 0.01 (***).",
    ]
    return "\n".join(lines)


def plot_cumulative_sse(forecasts, realized, path, maturity="xr2", benchmark=BENCHMARK):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench = forecasts[benchmark][maturity].dropna()
    y = realized.loc[bench.index, maturity]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in [n for names in PANELS.values() for n in names]:
        if name not in forecasts:
            continue
        fc = forecasts[name].loc[bench.index, maturity]
        diff = np.square(y - bench) - np.square(y - fc)
        style = (
            "-"
            if ("macro" in name or name.startswith("NN3") or "Ensem" in name)
            else "--"
        )
        ax.plot(diff.index, diff.cumsum(), style, label=name, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Cumulative SSE(benchmark) - SSE(model), {maturity[2:]}-year bond")
    ax.set_ylabel("Squared percent")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    from build_bbt_dataset import load_dataset

    data = load_dataset()
    realized = data["excess_returns"]
    forecasts = load_forecasts()
    if BENCHMARK not in forecasts:
        raise FileNotFoundError(
            f"Run `python src/forecast_oos.py --models {BENCHMARK}` first."
        )
    r2, pv = evaluate(forecasts, realized)
    order = [n for names in PANELS.values() for n in names if n in r2.index]
    order += [n for n in r2.index if n not in order]
    r2, pv = r2.loc[order], pv.loc[order]

    oos = forecasts[BENCHMARK].dropna(how="all").index
    oos_label = f"forecast origins {oos[0]:%Y:%m}--{oos[-1]:%Y:%m}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    r2.round(2).to_csv(OUTPUT_DIR / f"bbt_r2oos{RUN_SUFFIX}.csv")
    pv.round(4).to_csv(OUTPUT_DIR / f"bbt_r2oos_pvalues{RUN_SUFFIX}.csv")
    (OUTPUT_DIR / f"bbt_r2oos_table{RUN_SUFFIX}.tex").write_text(
        to_latex_table(r2, pv, oos_label)
    )
    plot_cumulative_sse(
        forecasts, realized, OUTPUT_DIR / f"bbt_cumulative_sse{RUN_SUFFIX}.png"
    )

    with pd.option_context("display.width", 140, "display.precision", 1):
        print(f"R2OOS (%), {oos_label}")
        print(r2)


if __name__ == "__main__":
    main()
