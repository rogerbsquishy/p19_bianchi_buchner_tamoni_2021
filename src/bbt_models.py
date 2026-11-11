"""Forecasting models for the recursive out-of-sample exercise.

Each model forecasts the vector of excess returns (one entry per maturity) for
a single forecast origin, given a training window. Inputs follow the
nomenclature of the provided replication code:

- ``Xm``: macro predictors (FRED-MD), T x K. ``None`` for yields-only models.
- ``Xf``: forward rates, T x 10 (``Xexog`` in the provided code).
- ``Y``: excess returns, T x M.

Hyperparameters are chosen on a validation sample made of the last 15% of the
training window, as in ``ElasticNet_Exog_Plain`` and the Keras
``validation_split=0.15`` used for the neural networks in ``NNFuncBib.py``.

Models expose

- ``tune(Xm, Xf, Y) -> params`` (only for models with ``retune_every``; the
  tree ensembles are re-tuned every ``HyperFreq`` months like the neural
  networks, and use fixed hyperparameters in between),
- ``fit_predict(Xm_train, Xf_train, Y_train, xm_test, xf_test, params)``
  returning an array of shape (M,).

The neural networks live in ``bbt_nn.py``.
"""

import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.model_selection import ParameterGrid, PredefinedSplit
from sklearn.preprocessing import StandardScaler

VAL_FRAC = 0.15


def n_train_rows(n, val_frac=VAL_FRAC):
    """Number of estimation rows when the last ``val_frac`` is held out.

    Matches ``ElasticNet_Exog_Plain`` in the provided code.

    >>> n_train_rows(100)
    85
    """
    return int(np.round(n * (1 - val_frac)))


def predefined_split(n, val_frac=VAL_FRAC):
    """Single train/validation split with the validation block at the end."""
    n_tr = n_train_rows(n, val_frac)
    test_fold = np.concatenate([np.full(n_tr, -1), np.full(n - n_tr, 0)])
    return PredefinedSplit(test_fold.tolist())


def design(Xm, Xf):
    """Stack macro and forward-rate predictors (forwards only if Xm is None)."""
    if Xm is None:
        return Xf
    return np.concatenate([Xm, Xf], axis=1)


def _as_2d(x):
    return np.asarray(x).reshape(1, -1)


class HistoricalMean:
    """Benchmark: the average excess return over the training window."""

    name = "HistoricalMean"
    uses_macro = False
    retune_every = None

    def fit_predict(self, Xm, Xf, Y, xm, xf, params=None):
        return Y.mean(axis=0)


class _ScaledLinearCV:
    """Standardize predictors, then fit one CV'd linear model per maturity."""

    retune_every = None

    def __init__(self, uses_macro):
        self.uses_macro = uses_macro
        suffix = "macro_fwd" if uses_macro else "fwd"
        self.name = f"{self.base_name}_{suffix}"

    def _make_model(self, cv):
        raise NotImplementedError

    def fit_predict(self, Xm, Xf, Y, xm, xf, params=None):
        # Scale macro and forward rates separately, as in the provided code
        f_scaler = StandardScaler().fit(Xf)
        Xtr, xte = f_scaler.transform(Xf), f_scaler.transform(_as_2d(xf))
        if self.uses_macro:
            m_scaler = StandardScaler().fit(Xm)
            Xtr = np.concatenate([m_scaler.transform(Xm), Xtr], axis=1)
            xte = np.concatenate([m_scaler.transform(_as_2d(xm)), xte], axis=1)

        ps = predefined_split(len(Xtr))
        pred = np.full(Y.shape[1], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for k in range(Y.shape[1]):
                model = self._make_model(ps).fit(Xtr, Y[:, k])
                pred[k] = model.predict(xte)[0]
        return pred


class ElasticNetModel(_ScaledLinearCV):
    """``ElasticNet_Exog_Plain`` from the provided code."""

    base_name = "ElasticNet"

    def _make_model(self, cv):
        return ElasticNetCV(
            cv=cv,
            max_iter=5000,
            n_jobs=1,
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            random_state=42,
        )


class LassoModel(_ScaledLinearCV):
    base_name = "Lasso"

    def _make_model(self, cv):
        return LassoCV(cv=cv, max_iter=5000, n_jobs=1, random_state=42)


class RidgeModel(_ScaledLinearCV):
    base_name = "Ridge"

    def _make_model(self, cv):
        return RidgeCV(alphas=np.logspace(-3, 5, 50), cv=cv)


def _val_mse(y_true, y_pred):
    return float(np.mean(np.square(y_true - y_pred)))


class PCARegression:
    """OLS on principal components, number of components chosen on validation.

    Yields-only: PCs of the forward rates (1 to ``max_fwd_pcs``).
    Macro + yields: PCs of the macro panel (1 to ``max_macro_pcs``) plus the
    first three PCs of the forward rates (level, slope, curvature).
    """

    retune_every = None

    def __init__(self, uses_macro, max_fwd_pcs=5, max_macro_pcs=10, n_fwd_pcs=3):
        self.uses_macro = uses_macro
        self.name = "PCA_macro_fwd" if uses_macro else "PCA_fwd"
        self.max_fwd_pcs = max_fwd_pcs
        self.max_macro_pcs = max_macro_pcs
        self.n_fwd_pcs = n_fwd_pcs

    @staticmethod
    def _pcs(X_fit, X_apply, k):
        scaler = StandardScaler().fit(X_fit)
        pca = PCA(n_components=k).fit(scaler.transform(X_fit))
        return [pca.transform(scaler.transform(x)) for x in X_apply]

    def _features(self, Xm_fit, Xf_fit, Xm_list, Xf_list, k):
        if self.uses_macro:
            fm = self._pcs(Xm_fit, Xm_list, k)
            ff = self._pcs(Xf_fit, Xf_list, self.n_fwd_pcs)
            return [np.concatenate([a, b], axis=1) for a, b in zip(fm, ff)]
        return self._pcs(Xf_fit, Xf_list, k)

    def fit_predict(self, Xm, Xf, Y, xm, xf, params=None):
        n_tr = n_train_rows(len(Y))
        k_max = self.max_macro_pcs if self.uses_macro else self.max_fwd_pcs
        k_max = min(k_max, (Xm if self.uses_macro else Xf).shape[1])
        best_k, best_loss = 1, np.inf
        for k in range(1, k_max + 1):
            Xm_tr = Xm[:n_tr] if self.uses_macro else None
            Xm_list = [Xm[:n_tr], Xm[n_tr:]] if self.uses_macro else None
            F_tr, F_val = self._features(
                Xm_tr, Xf[:n_tr], Xm_list, [Xf[:n_tr], Xf[n_tr:]], k
            )
            ols = LinearRegression().fit(F_tr, Y[:n_tr])
            loss = _val_mse(Y[n_tr:], ols.predict(F_val))
            if loss < best_loss:
                best_k, best_loss = k, loss

        Xm_list = [Xm, _as_2d(xm)] if self.uses_macro else None
        F_all, f_test = self._features(Xm, Xf, Xm_list, [Xf, _as_2d(xf)], best_k)
        return LinearRegression().fit(F_all, Y).predict(f_test)[0]


class PLSModel:
    """Partial least squares, one PLS1 regression per maturity."""

    retune_every = None

    def __init__(self, uses_macro, max_components=None):
        self.uses_macro = uses_macro
        self.name = "PLS_macro_fwd" if uses_macro else "PLS_fwd"
        self.max_components = max_components or (10 if uses_macro else 5)

    def fit_predict(self, Xm, Xf, Y, xm, xf, params=None):
        X = design(Xm if self.uses_macro else None, Xf)
        x = design(_as_2d(xm) if self.uses_macro else None, _as_2d(xf))
        n_tr = n_train_rows(len(Y))
        k_max = min(self.max_components, X.shape[1])
        pred = np.full(Y.shape[1], np.nan)
        for j in range(Y.shape[1]):
            best_k, best_loss = 1, np.inf
            for k in range(1, k_max + 1):
                pls = PLSRegression(n_components=k, scale=True)
                pls.fit(X[:n_tr], Y[:n_tr, j])
                loss = _val_mse(Y[n_tr:, j], pls.predict(X[n_tr:]).ravel())
                if loss < best_loss:
                    best_k, best_loss = k, loss
            pls = PLSRegression(n_components=best_k, scale=True).fit(X, Y[:, j])
            pred[j] = pls.predict(x).ravel()[0]
        return pred


class TreeEnsemble:
    """Random forest / extremely randomized trees (multi-output).

    Hyperparameters (tree depth and share of features per split) are tuned on
    the validation block every ``retune_every`` months and held fixed in
    between, mirroring the ``HyperFreq`` logic used for the neural networks.
    """

    def __init__(self, kind, uses_macro, n_estimators=300, retune_every=48, n_jobs=1):
        if kind not in ("RF", "ET"):
            raise ValueError("kind must be 'RF' or 'ET'")
        self.kind = kind
        self.uses_macro = uses_macro
        self.name = f"{kind}_{'macro_fwd' if uses_macro else 'fwd'}"
        self.n_estimators = n_estimators
        self.retune_every = retune_every
        self.n_jobs = n_jobs
        self.grid = list(
            ParameterGrid({"max_depth": [2, 4, 6, 8], "max_features": [0.1, 0.33, 1.0]})
        )

    def _make(self, params):
        cls = RandomForestRegressor if self.kind == "RF" else ExtraTreesRegressor
        return cls(
            n_estimators=self.n_estimators,
            random_state=0,
            n_jobs=self.n_jobs,
            **params,
        )

    def tune(self, Xm, Xf, Y):
        X = design(Xm if self.uses_macro else None, Xf)
        n_tr = n_train_rows(len(Y))
        losses = []
        for params in self.grid:
            model = self._make(params).fit(X[:n_tr], Y[:n_tr])
            losses.append(_val_mse(Y[n_tr:], model.predict(X[n_tr:])))
        return self.grid[int(np.argmin(losses))]

    def fit_predict(self, Xm, Xf, Y, xm, xf, params=None):
        if params is None:
            params = self.tune(Xm, Xf, Y)
        X = design(Xm if self.uses_macro else None, Xf)
        x = design(_as_2d(xm) if self.uses_macro else None, _as_2d(xf))
        return self._make(params).fit(X, Y).predict(x)[0]


def sklearn_models():
    """All non-neural-network models, keyed by name."""
    models = [HistoricalMean()]
    for uses_macro in (False, True):
        models += [
            PCARegression(uses_macro),
            PLSModel(uses_macro),
            RidgeModel(uses_macro),
            LassoModel(uses_macro),
            ElasticNetModel(uses_macro),
            TreeEnsemble("RF", uses_macro),
            TreeEnsemble("ET", uses_macro),
        ]
    return {m.name: m for m in models}
