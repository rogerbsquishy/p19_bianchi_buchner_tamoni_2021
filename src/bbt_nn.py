"""Neural networks of Bianchi, Buchner, and Tamoni (2021) in PyTorch.

This is a port of the Keras models in ``ProvidedReplicationCode/NNFuncBib.py``:

- ``NN3LayerExog`` (``ExogNet`` with ``archi=(32, 16, 8)``): a feed-forward
  network on the macro variables whose last hidden layer is concatenated with
  the forward rates, followed by dropout, batch normalization, and a linear
  output layer with one unit per maturity.
- ``NN1LayerEnsemExog`` (``GroupEnsembleNet`` with ``archi=(1,)``): one small
  network per McCracken-Ng macro group; the group outputs are concatenated
  with the forward rates, then dropout, batch normalization, and the output.
- ``NN1Layer_fwd`` (``ExogNet`` with no exogenous block): a one-hidden-layer
  network on the forward rates only, for the yields-only comparison.

Training follows the provided code: inputs scaled to [-1, 1] on the training
window, the last 15% of the window used for validation, SGD with Nesterov
momentum 0.9 and Keras-style learning-rate decay ``lr / (1 + decay * step)``,
batch size 32, up to 500 epochs with early stopping (patience 20) and the
best-validation weights restored. L1+L2 penalties (Keras ``l1_l2(l)``) apply
to the hidden-layer kernels and enter the validation loss, as in Keras.

Hyperparameters (dropout and penalty) are grid-searched at "refit" months.
In other months the network is warm-started from the weights saved at the
last refit month and re-trained on the expanded window, as the provided code
does with ``load_model``. One difference: after the grid search the provided
code trains the winning configuration a second time with the same seed; here
the winning network from the grid search is kept directly, which saves one
training run per refit month.
"""

import copy
import math

import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from torch import nn

VAL_FRAC = 0.15
NN_PARAM_GRID = {"dropout": [0.1, 0.3, 0.5], "l1l2": [0.01, 0.001]}


def _he_normal_(tensor, fan_in):
    """Keras ``he_normal``: truncated normal, std = sqrt(2 / fan_in)."""
    std = math.sqrt(2.0 / fan_in)
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def _dense(n_in, n_out, bias_he=True):
    layer = nn.Linear(n_in, n_out)
    _he_normal_(layer.weight, n_in)
    if bias_he:
        _he_normal_(layer.bias, n_out)
    else:
        nn.init.zeros_(layer.bias)
    return layer


class _Head(nn.Module):
    """Merge block: concat(features, forwards) -> dropout -> BN -> linear."""

    def __init__(self, n_in, n_out, dropout, bias_he):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Keras BatchNormalization defaults: momentum 0.99, epsilon 1e-3
        self.bn = nn.BatchNorm1d(n_in, momentum=0.01, eps=1e-3)
        self.out = _dense(n_in, n_out, bias_he=bias_he)

    def forward(self, z):
        return self.out(self.bn(self.dropout(z)))


class ExogNet(nn.Module):
    """Dense ReLU layers on ``x_main``, merged with ``x_exog`` at the end."""

    def __init__(self, n_main, n_exog, n_out, archi, dropout, l1l2):
        super().__init__()
        self.l1l2 = l1l2
        self.dropouts = nn.ModuleList()
        self.hidden = nn.ModuleList()
        n_in = n_main
        for width in archi:
            self.dropouts.append(nn.Dropout(dropout))
            self.hidden.append(_dense(n_in, width, bias_he=True))
            n_in = width
        self.head = _Head(n_in + n_exog, n_out, dropout, bias_he=True)

    def forward(self, x_main, x_exog=None):
        h = x_main
        for drop, dense in zip(self.dropouts, self.hidden):
            h = torch.relu(dense(drop(h)))
        if x_exog is not None:
            h = torch.cat([h, x_exog], dim=1)
        return self.head(h)

    def penalty(self):
        return self.l1l2 * sum(
            w.weight.abs().sum() + w.weight.square().sum() for w in self.hidden
        )


class GroupEnsembleNet(nn.Module):
    """One dense ReLU sub-network per macro group, merged with the forwards."""

    def __init__(self, group_sizes, n_exog, n_out, archi, dropout, l1l2):
        super().__init__()
        self.l1l2 = l1l2
        self.group_sizes = list(group_sizes)
        self.branches = nn.ModuleList()
        self.branch_dense = []
        for size in self.group_sizes:
            layers = []
            n_in = size
            for width in archi:
                dense = _dense(n_in, width, bias_he=False)
                layers += [nn.Dropout(dropout), dense, nn.ReLU()]
                self.branch_dense.append(dense)
                n_in = width
            self.branches.append(nn.Sequential(*layers))
        n_merge = archi[-1] * len(self.group_sizes) + n_exog
        self.head = _Head(n_merge, n_out, dropout, bias_he=False)

    def forward(self, x_main, x_exog):
        chunks = torch.split(x_main, self.group_sizes, dim=1)
        h = torch.cat([b(c) for b, c in zip(self.branches, chunks)], dim=1)
        return self.head(torch.cat([h, x_exog], dim=1))

    def penalty(self):
        return self.l1l2 * sum(
            d.weight.abs().sum() + d.weight.square().sum() for d in self.branch_dense
        )


def _batches(n, batch_size, rng):
    """Shuffled mini-batches; a trailing batch of one row joins the previous
    batch because batch normalization needs at least two rows."""
    perm = rng.permutation(n)
    batches = [perm[i : i + batch_size] for i in range(0, n, batch_size)]
    if len(batches) > 1 and len(batches[-1]) < 2:
        batches[-2] = np.concatenate([batches[-2], batches.pop()])
    return batches


def train_network(
    net,
    X_main,
    X_exog,
    Y,
    seed,
    lr,
    decay=0.01,
    momentum=0.9,
    epochs=500,
    patience=20,
    min_delta=1e-6,
    batch_size=32,
    val_frac=VAL_FRAC,
):
    """Train ``net`` in place; restore best-validation weights.

    Returns the minimum validation loss (MSE plus weight penalty).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = len(Y)
    n_tr = int(n * (1 - val_frac))  # Keras validation_split convention
    to_t = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float32)
    Xm, Xe, Yt = to_t(X_main), to_t(X_exog), to_t(Y)

    def exog(idx):
        return None if Xe is None else Xe[idx]

    tr_idx = np.arange(n_tr)
    va_idx = np.arange(n_tr, n)

    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: 1.0 / (1.0 + decay * step)
    )
    loss_fn = nn.MSELoss()

    best_loss, best_state, wait = np.inf, copy.deepcopy(net.state_dict()), 0
    for _ in range(epochs):
        net.train()
        for b in _batches(n_tr, batch_size, rng):
            idx = tr_idx[b]
            opt.zero_grad()
            loss = loss_fn(net(Xm[idx], exog(idx)), Yt[idx]) + net.penalty()
            loss.backward()
            opt.step()
            sched.step()

        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xm[va_idx], exog(va_idx)), Yt[va_idx]) + net.penalty()
            val = float(val)
        if val < best_loss:
            best_state = copy.deepcopy(net.state_dict())
        if val < best_loss - min_delta:
            wait = 0
        else:
            wait += 1
        best_loss = min(best_loss, val)
        if wait >= patience:
            break

    net.load_state_dict(best_state)
    net.eval()
    return best_loss


class NNSpec:
    """Configuration for one of the paper's network families."""

    def __init__(self, name, kind, archi, lr, uses_macro):
        self.name = name
        self.kind = kind  # "exog", "ensemble", or "fwd"
        self.archi = tuple(archi)
        self.lr = lr
        self.uses_macro = uses_macro

    def build(self, n_macro, n_fwd, n_out, group_sizes, dropout, l1l2):
        if self.kind == "exog":
            return ExogNet(n_macro, n_fwd, n_out, self.archi, dropout, l1l2)
        if self.kind == "ensemble":
            return GroupEnsembleNet(
                group_sizes, n_fwd, n_out, self.archi, dropout, l1l2
            )
        if self.kind == "fwd":
            return ExogNet(n_fwd, 0, n_out, self.archi, dropout, l1l2)
        raise ValueError(self.kind)


NN_SPECS = {
    "NN3LayerExog": NNSpec(
        "NN3LayerExog", "exog", (32, 16, 8), lr=0.01, uses_macro=True
    ),
    "NN1LayerEnsemExog": NNSpec(
        "NN1LayerEnsemExog", "ensemble", (1,), lr=0.015, uses_macro=True
    ),
    "NN1Layer_fwd": NNSpec("NN1Layer_fwd", "fwd", (3,), lr=0.01, uses_macro=False),
}


def _scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train)
    return scaler.transform(train), scaler.transform(test.reshape(1, -1))


def run_seed_path(
    spec_name,
    seed,
    Xm,
    Xf,
    Y,
    origins,
    train_ends,
    refit_flags,
    group_order=None,
    group_sizes=None,
    param_grid=NN_PARAM_GRID,
):
    """Recursive forecasts from one network seed (one of the ``nMC`` members).

    Parameters
    ----------
    origins : forecast-origin row indices.
    train_ends : exclusive end of the training window for each origin.
    refit_flags : whether to grid-search hyperparameters at each origin.
    group_order : column permutation putting macro series in group order
        (ensemble network only).

    Returns
    -------
    preds : array (len(origins), M) of forecasts.
    val_losses : array (len(origins),) of best validation losses.
    """
    torch.set_num_threads(1)
    spec = NN_SPECS[spec_name]
    if group_order is not None:
        Xm = Xm[:, group_order]
    M = Y.shape[1]
    preds = np.full((len(origins), M), np.nan)
    val_losses = np.full(len(origins), np.nan)
    base_state, base_params = None, None

    for j, (i, end, refit) in enumerate(zip(origins, train_ends, refit_flags)):
        xf_tr, xf_te = _scale(Xf[:end], Xf[i])
        if spec.uses_macro:
            xm_tr, xm_te = _scale(Xm[:end], Xm[i])
            main_tr, main_te, ex_tr, ex_te = xm_tr, xm_te, xf_tr, xf_te
        else:
            main_tr, main_te, ex_tr, ex_te = xf_tr, xf_te, None, None
        y_tr = Y[:end]

        def build(params):
            return spec.build(
                Xm.shape[1] if spec.uses_macro else 0,
                Xf.shape[1],
                M,
                group_sizes,
                params["dropout"],
                params["l1l2"],
            )

        if refit or base_state is None:
            best = (np.inf, None, None)
            for params in ParameterGrid(param_grid):
                torch.manual_seed(seed)
                net = build(params)
                loss = train_network(net, main_tr, ex_tr, y_tr, seed, spec.lr)
                if loss < best[0]:
                    best = (loss, net, params)
            loss, net, base_params = best
            base_state = copy.deepcopy(net.state_dict())
        else:
            net = build(base_params)
            net.load_state_dict(base_state)
            loss = train_network(net, main_tr, ex_tr, y_tr, seed, spec.lr)

        with torch.no_grad():
            x_main = torch.as_tensor(main_te, dtype=torch.float32)
            x_ex = (
                None if ex_te is None else torch.as_tensor(ex_te, dtype=torch.float32)
            )
            preds[j] = net(x_main, x_ex).numpy()[0]
        val_losses[j] = loss
    return preds, val_losses


def aggregate_members(preds, val_losses, n_avg):
    """Average the ``n_avg`` members with the lowest validation loss.

    Parameters
    ----------
    preds : array (n_members, n_origins, M)
    val_losses : array (n_members, n_origins)

    >>> p = np.array([[[1.0]], [[2.0]], [[9.0]]])
    >>> v = np.array([[0.1], [0.2], [0.9]])
    >>> aggregate_members(p, v, 2)
    array([[1.5]])
    """
    order = np.argsort(val_losses, axis=0)[:n_avg]  # (n_avg, n_origins)
    picked = np.take_along_axis(preds, order[:, :, None], axis=0)
    return picked.mean(axis=0)
