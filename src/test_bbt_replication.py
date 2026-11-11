"""Tests for the Bianchi, Buchner, and Tamoni (2021) replication pipeline.

All tests use synthetic data, so they run without network access.
"""

import numpy as np
import pandas as pd
import pytest

from bbt_models import ElasticNetModel, HistoricalMean, PCARegression, TreeEnsemble
from build_bbt_dataset import (
    build_dataset,
    excess_returns,
    forward_rates,
    monthly_yields,
    transform_series,
)
from evaluate_oos import clark_west_pvalue, r2_oos
from forecast_oos import forecast_schedule, run_sklearn_model


def _fake_daily_gsw(start="1970-01-01", end="2020-12-31", seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    level = 5 + np.cumsum(rng.normal(0, 0.02, len(days)))
    cols = {}
    for n in range(1, 31):
        cols["SVENY" + str(n).zfill(2)] = level + 0.1 * np.log(n)
    return pd.DataFrame(cols, index=days)


def _fake_fred_md(start="1959-01-01", end="2020-12-01", seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="MS")
    raw = pd.DataFrame(
        {
            "INDPRO": np.exp(np.cumsum(rng.normal(0.002, 0.01, len(idx)))),
            "UNRATE": 5 + np.cumsum(rng.normal(0, 0.1, len(idx))),
            "HOUST": np.exp(7 + rng.normal(0, 0.1, len(idx))),
            "LATE_SERIES": np.r_[np.full(300, np.nan), np.ones(len(idx) - 300)],
        },
        index=idx,
    )
    info = pd.DataFrame(
        {"tcode": [5, 2, 4, 1], "group": [1.0, 2.0, 3.0, 4.0], "source": "test"},
        index=raw.columns,
    )
    return raw, info


def test_monthly_yields_takes_month_end():
    daily = _fake_daily_gsw("2000-01-01", "2000-03-31")
    y = monthly_yields(daily)
    assert list(y.index.strftime("%Y-%m-%d")) == [
        "2000-01-31",
        "2000-02-29",
        "2000-03-31",
    ]
    assert y.loc["2000-01-31", "y1"] == daily.loc["2000-01-31", "SVENY01"]


def test_forward_rates_flat_curve():
    y = pd.DataFrame({f"y{n}": [4.0] for n in range(1, 11)})
    f = forward_rates(y)
    assert np.allclose(f.to_numpy(), 4.0)


def test_excess_returns_formula():
    idx = pd.date_range("2000-01-31", periods=25, freq="ME")
    rng = np.random.default_rng(0)
    y = pd.DataFrame({f"y{n}": rng.uniform(1, 6, 25) for n in range(1, 11)}, index=idx)
    xr = excess_returns(y, maturities=(5,))
    t = 3
    expected = -4 * y["y4"].iloc[t + 12] + 5 * y["y5"].iloc[t] - y["y1"].iloc[t]
    assert xr["xr5"].iloc[t] == pytest.approx(expected)
    assert xr["xr5"].iloc[-12:].isna().all()


def test_transform_series_codes():
    s = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0])
    assert transform_series(s, 1).equals(s)
    assert np.allclose(transform_series(s, 3).dropna(), [1.0, 2.0, 4.0])
    assert np.allclose(transform_series(s, 6).dropna(), 0.0)
    with pytest.raises(ValueError):
        transform_series(s, 9)


def test_build_dataset_alignment_and_balancing():
    raw, info = _fake_fred_md()
    data = build_dataset(_fake_daily_gsw(), raw, info, "1971-08", "2017-12")
    fwd, macro, xr = data["forwards"], data["macro"], data["excess_returns"]
    assert fwd.index[0] == pd.Timestamp("1971-08-31")
    assert fwd.index[-1] == pd.Timestamp("2017-12-31")
    assert fwd.index.equals(macro.index) and fwd.index.equals(xr.index)
    assert list(xr.columns) == ["xr2", "xr3", "xr4", "xr5", "xr7", "xr10"]
    assert fwd.shape[1] == 10
    # The series that starts late is dropped to keep the panel balanced
    assert "LATE_SERIES" not in macro.columns
    assert not macro.isna().any().any()
    assert set(data["macro_groups"]["group"]) == {1, 2, 3}


def test_build_dataset_rejects_end_without_future_returns():
    raw, info = _fake_fred_md()
    with pytest.raises(ValueError, match="Missing yields/returns"):
        build_dataset(_fake_daily_gsw(), raw, info, "1971-08", "2020-12")


def test_forecast_schedule_has_no_lookahead():
    dates = pd.date_range("1980-01-31", periods=200, freq="ME")
    origins, ends, _ = forecast_schedule(dates, "1990-01", horizon=12)
    # Last training row (end - 1) has its 12-month return realized by the origin
    assert np.all((ends - 1) + 12 <= origins)
    assert np.all(origins - (ends - 1) == 12)


def test_r2_oos_and_clark_west():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=600)
    y = signal + rng.normal(scale=0.5, size=600)
    bench = np.zeros_like(y)
    assert r2_oos(y, signal, bench) > 0.5
    assert clark_west_pvalue(y, signal, bench) < 0.01
    assert r2_oos(y, bench, bench) == pytest.approx(0.0)
    noise = rng.normal(size=600)
    assert r2_oos(y, noise, bench) < 0


def _synthetic_predictive_data(T=220, seed=0):
    rng = np.random.default_rng(seed)
    Xf = rng.normal(size=(T, 10))
    Xm = rng.normal(size=(T, 6))
    beta = np.zeros((10, 2))
    beta[0] = [1.0, 2.0]
    Y = Xf @ beta + 0.5 * Xm[:, [0]] + 0.3 * rng.normal(size=(T, 2))
    return Xm, Xf, Y


def test_elastic_net_beats_mean_on_predictable_data():
    Xm, Xf, Y = _synthetic_predictive_data()
    dates = pd.date_range("2000-01-31", periods=len(Y), freq="ME")
    origins, ends, refit = forecast_schedule(dates, "2012-01", horizon=12)
    origins, ends, refit = origins[:24], ends[:24], refit[:24]
    enet = run_sklearn_model(
        ElasticNetModel(True), Xm, Xf, Y, origins, ends, refit, n_jobs=1
    )
    mean = run_sklearn_model(
        HistoricalMean(), Xm, Xf, Y, origins, ends, refit, n_jobs=1
    )
    assert enet.shape == (24, 2)
    for k in range(2):
        assert r2_oos(Y[origins, k], enet[:, k], mean[:, k]) > 0.5


def test_pca_and_trees_run():
    Xm, Xf, Y = _synthetic_predictive_data(T=150)
    for model in (
        PCARegression(False),
        PCARegression(True),
        TreeEnsemble("ET", True, n_estimators=20),
    ):
        pred = model.fit_predict(Xm[:120], Xf[:120], Y[:120], Xm[130], Xf[130])
        assert pred.shape == (2,) and np.all(np.isfinite(pred))


def test_neural_networks_run():
    torch = pytest.importorskip("torch")  # noqa: F841
    from bbt_nn import NN_SPECS, aggregate_members, run_seed_path

    Xm, Xf, Y = _synthetic_predictive_data(T=120)
    groups = np.array([1, 1, 2, 2, 3, 3])
    origins = np.array([110, 111, 112])
    ends = origins - 11
    refit = np.array([True, False, False])
    grid = {"dropout": [0.1], "l1l2": [0.001]}
    for name in NN_SPECS:
        out = [
            run_seed_path(
                name,
                seed,
                Xm,
                Xf,
                Y,
                origins,
                ends,
                refit,
                group_order=np.argsort(groups),
                group_sizes=[2, 2, 2],
                param_grid=grid,
            )
            for seed in range(2)
        ]
        preds = np.stack([o[0] for o in out])
        losses = np.stack([o[1] for o in out])
        assert preds.shape == (2, 3, 2) and np.all(np.isfinite(preds))
        agg = aggregate_members(preds, losses, 1)
        assert agg.shape == (3, 2)
