"""Build the forecasting dataset of Bianchi, Buchner, and Tamoni (2021).

All variables are monthly and indexed by the *forecast origin* ``t`` (the
month-end at which the predictors are observed):

- Forward rates ``f^(n)_t`` for n = 1, ..., 10 years, from Gurkaynak, Sack, and
  Wright (2007) zero-coupon yields (``Xexog`` in the provided code).
- Macro predictors: the FRED-MD panel after its stationarity transformations
  (``X`` in the provided code), and the McCracken-Ng group of each series
  (``A`` in the provided code).
- Targets: annual excess log returns ``rx^(n)_{t+12}`` for n = 2, 3, 4, 5, 7,
  and 10 years (``Y`` in the provided code). Row ``t`` holds the return from
  ``t`` to ``t + 12``.

With continuously compounded yields ``y^(n)_t`` in percent, log prices are
``p^(n)_t = -n y^(n)_t``, so

    f^(n)_t           = p^(n-1)_t - p^(n)_t
    rx^(n)_{t+12}     = p^(n-1)_{t+12} - p^(n)_t - y^(1)_t

All quantities are in percent.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
SAMPLE_START = config("BBT_SAMPLE_START", default="1971-08")
SAMPLE_END = config("BBT_SAMPLE_END", default="2017-12")
MATURITIES = (2, 3, 4, 5, 7, 10)
N_FORWARDS = 10
HORIZON = 12


def monthly_yields(daily, max_maturity=N_FORWARDS):
    """Month-end GSW zero yields ``y1, ..., y{max_maturity}`` (percent).

    Takes the last available daily observation in each month. The result is
    indexed by month-end timestamps on a complete monthly grid.
    """
    cols = ["SVENY" + str(n).zfill(2) for n in range(1, max_maturity + 1)]
    df = daily[cols].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().groupby(df.index.to_period("M")).last()
    df = df.reindex(pd.period_range(df.index.min(), df.index.max(), freq="M"))
    df.index = df.index.to_timestamp(how="end").normalize()
    df.index.name = "date"
    df.columns = [f"y{n}" for n in range(1, max_maturity + 1)]
    return df


def forward_rates(yields):
    """One-year forward rates ``f1, ..., fN`` from zero yields ``y1, ..., yN``.

    >>> y = pd.DataFrame({"y1": [2.0], "y2": [3.0], "y3": [3.5]})
    >>> forward_rates(y).round(6).iloc[0].to_dict()
    {'f1': 2.0, 'f2': 4.0, 'f3': 4.5}
    """
    n_max = yields.shape[1]
    fwd = pd.DataFrame(index=yields.index)
    fwd["f1"] = yields["y1"]
    for n in range(2, n_max + 1):
        fwd[f"f{n}"] = n * yields[f"y{n}"] - (n - 1) * yields[f"y{n - 1}"]
    return fwd


def excess_returns(yields, maturities=MATURITIES, horizon=HORIZON):
    """Annual excess log returns indexed by the start of the holding period.

    Requires ``yields`` on a complete monthly grid so that shifting by
    ``horizon`` rows moves exactly ``horizon`` months ahead.

    >>> idx = pd.date_range("2000-01-31", periods=13, freq="ME")
    >>> y = pd.DataFrame({"y1": 1.0, "y2": 2.0}, index=idx)
    >>> float(excess_returns(y, maturities=(2,)).iloc[0, 0])
    2.0
    """
    xr = pd.DataFrame(index=yields.index)
    for n in maturities:
        p_n_t = -n * yields[f"y{n}"]
        p_nm1_tph = -(n - 1) * yields[f"y{n - 1}"].shift(-horizon)
        xr[f"xr{n}"] = p_nm1_tph - p_n_t - yields["y1"]
    return xr


def transform_series(x, tcode):
    """Apply a FRED-MD transformation code to a series.

    >>> s = pd.Series([1.0, 2.0, 4.0, 8.0])
    >>> transform_series(s, 2).tolist()[1:]
    [1.0, 2.0, 4.0]
    >>> np.round(transform_series(s, 5).tolist()[1:], 4).tolist()
    [0.6931, 0.6931, 0.6931]
    """
    x = x.astype(float)
    tcode = int(tcode)
    if tcode in (4, 5, 6):
        x = x.where(x > 0)
    if tcode == 1:
        return x
    if tcode == 2:
        return x.diff()
    if tcode == 3:
        return x.diff().diff()
    if tcode == 4:
        return np.log(x)
    if tcode == 5:
        return np.log(x).diff()
    if tcode == 6:
        return np.log(x).diff().diff()
    if tcode == 7:
        return (x / x.shift(1) - 1).diff()
    raise ValueError(f"Unknown transformation code {tcode}")


def transform_fred_md(raw, info):
    """Transform FRED-MD levels and index them by month-end."""
    out = pd.DataFrame(
        {c: transform_series(raw[c], info.loc[c, "tcode"]) for c in raw.columns}
    )
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp(how="end")
    out.index = out.index.normalize()
    out.index.name = "date"
    return out


def build_dataset(
    daily_yields,
    fred_md_raw,
    fred_md_info,
    sample_start=SAMPLE_START,
    sample_end=SAMPLE_END,
    maturities=MATURITIES,
    horizon=HORIZON,
):
    """Align forward rates, macro predictors, and excess returns.

    Macro series with any missing value inside the sample, or without a
    McCracken-Ng group, are dropped so that the panel is balanced.

    Returns
    -------
    dict with DataFrames ``forwards`` (T x 10), ``macro`` (T x K),
    ``excess_returns`` (T x len(maturities)), and ``macro_groups`` (K x 2).
    """
    yields = monthly_yields(daily_yields, max_maturity=max(N_FORWARDS, *maturities))
    fwd = forward_rates(yields)[[f"f{n}" for n in range(1, N_FORWARDS + 1)]]
    xr = excess_returns(yields, maturities=maturities, horizon=horizon)

    macro = transform_fred_md(fred_md_raw, fred_md_info)
    groups = fred_md_info["group"].dropna()
    macro = macro[[c for c in macro.columns if c in groups.index]]

    start = pd.Period(sample_start, freq="M").to_timestamp(how="end").normalize()
    end = pd.Period(sample_end, freq="M").to_timestamp(how="end").normalize()
    dates = pd.date_range(start, end, freq="ME")
    fwd, xr, macro = (df.reindex(dates) for df in (fwd, xr, macro))

    if fwd.isna().any().any() or xr.isna().any().any():
        bad = dates[fwd.isna().any(axis=1) | xr.isna().any(axis=1)]
        raise ValueError(
            f"Missing yields/returns in sample for {len(bad)} months, "
            f"first {bad[0]:%Y-%m}. Is BBT_SAMPLE_END at least {horizon} months "
            "before the end of the yield data?"
        )

    complete = macro.columns[macro.notna().all()]
    dropped = sorted(set(macro.columns) - set(complete))
    if dropped:
        print(f"Dropping {len(dropped)} macro series with gaps in sample: {dropped}")
    macro = macro[complete]

    macro_groups = pd.DataFrame(
        {"group": groups.loc[macro.columns].astype(int)}, index=macro.columns
    )
    macro_groups["tcode"] = fred_md_info.loc[macro.columns, "tcode"].astype(int)
    macro_groups.index.name = "series"

    for df in (fwd, xr, macro):
        df.index.name = "date"
    return {
        "forwards": fwd,
        "macro": macro,
        "excess_returns": xr,
        "macro_groups": macro_groups,
    }


DATASET_FILES = {
    "forwards": "bbt_forward_rates.parquet",
    "macro": "bbt_macro.parquet",
    "excess_returns": "bbt_excess_returns.parquet",
    "macro_groups": "bbt_macro_groups.parquet",
}


def save_dataset(data, data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for key, fname in DATASET_FILES.items():
        data[key].to_parquet(data_dir / fname)


def load_dataset(data_dir=DATA_DIR):
    """Load the aligned replication dataset built by this module."""
    data_dir = Path(data_dir)
    return {key: pd.read_parquet(data_dir / f) for key, f in DATASET_FILES.items()}


if __name__ == "__main__":
    from load_fed_yield_curve import load_fed_yield_curve
    from pull_fred_md import load_fred_md

    daily = load_fed_yield_curve(DATA_DIR)
    raw, info = load_fred_md(DATA_DIR)
    data = build_dataset(daily, raw, info)
    save_dataset(data)
    T = len(data["forwards"])
    K = data["macro"].shape[1]
    n_groups = data["macro_groups"]["group"].nunique()
    print(
        f"Saved dataset: {T} months ({data['forwards'].index[0]:%Y-%m} to "
        f"{data['forwards'].index[-1]:%Y-%m}), {K} macro series in {n_groups} groups."
    )
