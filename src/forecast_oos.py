"""Recursive (expanding-window) out-of-sample forecasts of bond excess returns.

Python port of ``ProvidedReplicationCode/ModelComparison_Rolling.py``.

At each forecast origin ``t`` from ``BBT_OOS_START`` onward, every model is
estimated on data up to ``t`` and forecasts the excess returns from ``t`` to
``t + 12``. Neural networks are trained ``BBT_NN_NMC`` times with different
seeds and the ``BBT_NN_NAVG`` members with the lowest validation loss are
averaged (``nMC`` / ``nAvg`` in the provided code). Hyperparameters of the
networks and tree ensembles are re-tuned every ``BBT_HYPER_FREQ`` months.

Look-ahead and the training window
----------------------------------
Row ``s`` of the target is the return from ``s`` to ``s + 12``, which is only
known at ``s + 12``. With ``BBT_NO_LOOKAHEAD=True`` (default), the training
window at origin ``t`` ends at row ``t - 12`` so that every training target has
been realized by ``t``, and the historical-mean benchmark uses the same rows.
``BBT_NO_LOOKAHEAD=False`` reproduces the provided code literally: it trains on
rows ``0, ..., t - 1``, whose last 11 targets are not yet observed at ``t``.

Usage
-----
    python src/forecast_oos.py                       # all models
    python src/forecast_oos.py --models ElasticNet_macro_fwd NN3LayerExog
    python src/forecast_oos.py --BBT_NN_NMC=8 --BBT_NN_NAVG=4  # quicker NNs

Forecasts are saved to ``DATA_DIR/oos_forecasts/<model>.parquet``
(``oos_forecasts_lookahead`` when ``BBT_NO_LOOKAHEAD=False``).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bbt_models import sklearn_models
from build_bbt_dataset import HORIZON, load_dataset
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OOS_START = config("BBT_OOS_START", default="1989-01")
HYPER_FREQ = config("BBT_HYPER_FREQ", default=48, cast=int)
NN_NMC = config("BBT_NN_NMC", default=100, cast=int)
NN_NAVG = config("BBT_NN_NAVG", default=10, cast=int)
NO_LOOKAHEAD = config(
    "BBT_NO_LOOKAHEAD",
    default=True,
    cast=lambda v: str(v).lower() in ("1", "true", "yes"),
)
N_JOBS = config("BBT_N_JOBS", default=-1, cast=int)
# Keep forecasts from the provided code's timing separate from the default run
RUN_SUFFIX = "" if NO_LOOKAHEAD else "_lookahead"
FORECAST_DIR = DATA_DIR / f"oos_forecasts{RUN_SUFFIX}"

NN_MODEL_NAMES = ["NN1Layer_fwd", "NN3LayerExog", "NN1LayerEnsemExog"]


def forecast_schedule(
    dates,
    oos_start=OOS_START,
    horizon=HORIZON,
    no_lookahead=NO_LOOKAHEAD,
    hyper_freq=HYPER_FREQ,
):
    """Forecast origins, training-window ends, and re-tuning flags.

    ``train_ends[j]`` is the exclusive end row of the training window for
    origin ``origins[j]``. Re-tuning happens at the first origin and whenever
    the 1-based counter is a multiple of ``hyper_freq``, as in the provided code.

    >>> dates = pd.date_range("2000-01-31", periods=30, freq="ME")
    >>> o, e, r = forecast_schedule(dates, "2001-06", horizon=12, hyper_freq=4)
    >>> int(o[0]), int(e[0]), r[:5].tolist()
    (17, 6, [True, False, False, True, False])
    >>> o, e, r = forecast_schedule(dates, "2001-06", horizon=12, no_lookahead=False)
    >>> int(e[0])
    17
    """
    start = pd.Period(oos_start, freq="M")
    periods = pd.DatetimeIndex(dates).to_period("M")
    first = int(np.argmax(periods >= start))
    if periods[first] < start:
        raise ValueError(f"OOS start {oos_start} is after the end of the sample")
    origins = np.arange(first, len(dates))
    gap = horizon - 1 if no_lookahead else 0
    train_ends = origins - gap
    counter = np.arange(1, len(origins) + 1)
    refit = (counter == 1) | (counter % hyper_freq == 0)
    return origins, train_ends, refit


def _sk_one(model, Xm, Xf, Y, i, end, params):
    return model.fit_predict(Xm[:end], Xf[:end], Y[:end], Xm[i], Xf[i], params)


def run_sklearn_model(model, Xm, Xf, Y, origins, train_ends, refit, n_jobs=N_JOBS):
    """Forecasts (len(origins) x M) for a non-neural-network model."""
    params = [None] * len(origins)
    if getattr(model, "retune_every", None):
        tune_rows = np.flatnonzero(refit)
        tuned = Parallel(n_jobs=n_jobs)(
            delayed(model.tune)(
                Xm[: train_ends[j]], Xf[: train_ends[j]], Y[: train_ends[j]]
            )
            for j in tune_rows
        )
        current = None
        lookup = dict(zip(tune_rows, tuned))
        for j in range(len(origins)):
            current = lookup.get(j, current)
            params[j] = current
    preds = Parallel(n_jobs=n_jobs)(
        delayed(_sk_one)(model, Xm, Xf, Y, i, end, p)
        for i, end, p in zip(origins, train_ends, params)
    )
    return np.vstack(preds)


def run_nn_model(
    name,
    Xm,
    Xf,
    Y,
    groups,
    origins,
    train_ends,
    refit,
    n_mc=NN_NMC,
    n_avg=NN_NAVG,
    n_jobs=N_JOBS,
):
    """Ensemble forecasts for a neural network, parallelized over seeds."""
    from bbt_nn import aggregate_members, run_seed_path

    group_order = np.argsort(groups, kind="stable")
    group_sizes = np.unique(groups[group_order], return_counts=True)[1].tolist()
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_seed_path)(
            name,
            seed,
            Xm,
            Xf,
            Y,
            origins,
            train_ends,
            refit,
            group_order=group_order,
            group_sizes=group_sizes,
        )
        for seed in range(n_mc)
    )
    member_preds = np.stack([o[0] for o in out])  # (n_mc, n_origins, M)
    member_val = np.stack([o[1] for o in out])  # (n_mc, n_origins)
    return (
        aggregate_members(member_preds, member_val, min(n_avg, n_mc)),
        member_preds,
        member_val,
    )


def save_forecasts(
    name, preds, dates, origins, columns, extra=None, out_dir=FORECAST_DIR
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(np.nan, index=dates, columns=columns)
    df.iloc[origins] = preds
    df.index.name = "date"
    df.to_parquet(out_dir / f"{name}.parquet")
    if extra is not None:
        np.savez_compressed(out_dir / f"{name}_members.npz", **extra)
    return df


def main(model_names=None):
    data = load_dataset()
    Xf = data["forwards"].to_numpy()
    Xm = data["macro"].to_numpy()
    Y = data["excess_returns"].to_numpy()
    groups = data["macro_groups"].loc[data["macro"].columns, "group"].to_numpy()
    dates = data["forwards"].index
    columns = data["excess_returns"].columns

    origins, train_ends, refit = forecast_schedule(dates)
    print(
        f"{len(origins)} forecast origins {dates[origins[0]]:%Y-%m} to "
        f"{dates[origins[-1]]:%Y-%m}; {Xm.shape[1]} macro series; "
        f"no_lookahead={NO_LOOKAHEAD}"
    )

    sk = sklearn_models()
    all_names = list(sk) + NN_MODEL_NAMES
    model_names = model_names or all_names
    unknown = set(model_names) - set(all_names)
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}. Choose from {all_names}")

    for name in model_names:
        start = time.time()
        if name in sk:
            preds = run_sklearn_model(sk[name], Xm, Xf, Y, origins, train_ends, refit)
            save_forecasts(name, preds, dates, origins, columns)
        else:
            preds, mp, mv = run_nn_model(
                name, Xm, Xf, Y, groups, origins, train_ends, refit
            )
            save_forecasts(
                name,
                preds,
                dates,
                origins,
                columns,
                extra={"member_preds": mp, "member_val_loss": mv, "origins": origins},
            )
        print(f"{name}: done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="*", default=None)
    args, _ = parser.parse_known_args()  # ALL_CAPS --VAR=value flags go to settings
    main(args.models)
