"""Pull the FRED-MD monthly macro panel (McCracken and Ng, 2016).

Bianchi, Buchner, and Tamoni (2021) use FRED-MD as their set of macroeconomic
predictors. The data are obtained in the following order of preference:

1. A FRED-MD CSV that you downloaded by hand and saved as
   ``data_manual/fred_md.csv`` (any vintage, e.g. ``2019-02.csv`` renamed).
   This is the most faithful option. Download vintages from
   https://www.stlouisfed.org/research/economists/mccracken/fred-databases
2. The official download URL given by the ``FRED_MD_URL`` setting (defaults to
   the current vintage). The St. Louis Fed's file server rejects many scripted
   requests, so this step may fail.
3. A fallback panel rebuilt series-by-series from FRED
   (https://fred.stlouisfed.org/graph/fredgraph.csv) using the FRED series ids
   and transformation codes in ``fred_md_metadata.py``. Series that FRED does
   not host with sufficient history (e.g. the S&P 500 block, the help-wanted
   index, and the commercial paper rate) are missing in this fallback.

Two files are written to ``DATA_DIR``:

- ``fred_md_raw.parquet``: untransformed series, monthly, indexed by month start.
- ``fred_md_tcodes.parquet``: transformation code, group, and data source for
  each series.
"""

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fred_md_metadata import FRED_MD_METADATA
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))
FRED_MD_URL = config(
    "FRED_MD_URL",
    default="https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv",
)
FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def parse_fred_md_csv(text_or_path):
    """Parse a FRED-MD CSV into (raw levels, transformation codes).

    The FRED-MD file has a header row of series ids, a second row
    ("Transform:") holding the transformation codes, then one row per month
    with dates formatted as m/d/yyyy.

    >>> csv = (
    ...     "sasdate,RPI,UNRATE\\n"
    ...     "Transform:,5,2\\n"
    ...     "1/1/1959,2437.296,6.0\\n"
    ...     "2/1/1959,2446.902,5.9\\n"
    ... )
    >>> raw, tcodes = parse_fred_md_csv(io.StringIO(csv))
    >>> list(raw.columns), tcodes.to_dict()
    (['RPI', 'UNRATE'], {'RPI': 5, 'UNRATE': 2})
    >>> str(raw.index[1].date())
    '1959-02-01'
    """
    df = pd.read_csv(text_or_path)
    first_col = df.columns[0]
    is_tcode_row = df[first_col].astype(str).str.lower().str.startswith("transform")
    tcodes = df.loc[is_tcode_row].drop(columns=first_col).iloc[0].astype(int)
    data = df.loc[~is_tcode_row].dropna(subset=[first_col])
    data.index = pd.to_datetime(data[first_col], format="%m/%d/%Y")
    data.index.name = "date"
    data = data.drop(columns=first_col).apply(pd.to_numeric, errors="coerce")
    data = data.dropna(how="all")
    return data, tcodes


def _fetch_fredgraph(series_id, timeout=60, retries=3):
    """Download one series from FRED's public CSV endpoint as a monthly Series.

    One series per request: when several series with different native
    frequencies are requested together, FRED returns a ZIP archive instead of
    a CSV. Weekly or daily series are averaged within each month.
    """
    for attempt in range(retries):
        try:
            response = requests.get(
                FREDGRAPH_URL, params={"id": series_id}, timeout=timeout
            )
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    text = response.text
    if "csv" not in response.headers.get("content-type", "csv"):
        raise ValueError(f"FRED returned non-CSV content for {series_id}")
    df = pd.read_csv(io.StringIO(text), na_values=".")
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    s = s.groupby(s.index.to_period("M")).mean()
    s.index = s.index.to_timestamp()
    s.index.name = "date"
    s.name = series_id
    return s


def pull_fred_md_from_fred():
    """Rebuild an approximation of FRED-MD from individual FRED series.

    Returns
    -------
    raw : DataFrame of monthly levels, columns named by FRED-MD id.
    tcodes : Series of transformation codes indexed by FRED-MD id.
    """
    meta = FRED_MD_METADATA.dropna(subset=["fred_id"])
    meta = meta[~meta["fred_id"].duplicated()]

    series = []
    for sid in meta["fred_id"]:
        try:
            series.append(_fetch_fredgraph(sid))
        except (requests.RequestException, ValueError, pd.errors.ParserError) as e:
            print(f"Could not pull {sid} from FRED: {e}")
    fred = pd.concat(series, axis=1)

    # PI is only needed to construct CONSPI (nonrevolving credit / personal income)
    pi = _fetch_fredgraph("PI")

    rename = {fid: mdid for mdid, fid in meta["fred_id"].items()}
    raw = fred.rename(columns=rename)
    raw["CONSPI"] = raw["NONREVSL"] / pi.reindex(raw.index)
    raw = raw.sort_index()

    tcodes = FRED_MD_METADATA.loc[raw.columns, "tcode"].astype(int)
    return raw, tcodes


def pull_fred_md(url=FRED_MD_URL, manual_dir=MANUAL_DATA_DIR):
    """Get the FRED-MD panel, trying manual file, official URL, then FRED.

    Returns
    -------
    raw : DataFrame of monthly levels.
    info : DataFrame with columns ``tcode``, ``group`` and ``source``.
    """
    manual_path = Path(manual_dir) / "fred_md.csv"
    if manual_path.exists():
        print(f"Using manually downloaded FRED-MD file: {manual_path}")
        raw, tcodes = parse_fred_md_csv(manual_path)
        source = f"manual:{manual_path.name}"
    else:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            raw, tcodes = parse_fred_md_csv(io.StringIO(response.text))
            source = f"url:{url}"
            print(f"Downloaded FRED-MD from {url}")
        except (requests.RequestException, ValueError, IndexError) as e:
            print(
                f"Could not download FRED-MD from {url} ({e}).\n"
                "Falling back to rebuilding the panel from individual FRED series. "
                f"For the official data, save a FRED-MD CSV to {manual_path}."
            )
            raw, tcodes = pull_fred_md_from_fred()
            source = "fred_rebuild"

    groups = pd.Series(
        [FRED_MD_METADATA["group"].get(c, np.nan) for c in raw.columns],
        index=raw.columns,
        dtype="float",
    )
    unmapped = groups[groups.isna()].index.tolist()
    if unmapped:
        print(
            f"Warning: no McCracken-Ng group for {unmapped}; "
            "they are dropped from the macro panel."
        )
    info = pd.DataFrame({"tcode": tcodes, "group": groups, "source": source})
    info.index.name = "series"
    return raw, info


def load_fred_md(data_dir=DATA_DIR):
    """Load the cached raw FRED-MD panel and its transformation info."""
    data_dir = Path(data_dir)
    raw = pd.read_parquet(data_dir / "fred_md_raw.parquet")
    info = pd.read_parquet(data_dir / "fred_md_tcodes.parquet")
    return raw, info


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw, info = pull_fred_md()
    raw.to_parquet(DATA_DIR / "fred_md_raw.parquet")
    info.to_parquet(DATA_DIR / "fred_md_tcodes.parquet")
    print(
        f"Saved {raw.shape[1]} series, {raw.index.min():%Y-%m} to "
        f"{raw.index.max():%Y-%m} (source: {info['source'].iloc[0]})"
    )
