"""Checks of the Python bindings against the Python port (erweinstein/quantile_regression).

    pip install maturin && (cd python && maturin develop --release)   # or: maturin build, pip install the wheel
    QR_PYTHON=/path/to/quantile_regression python -m pytest python/tests

The point estimates should agree to rounding (about 1e-12 or better); the bootstraps draw their weights from
different generators (Xoshiro256++ here, numpy's PCG64 there), so only their shapes are checked.

Eliot R. Weinstein
eliotw@uchicago.edu
erweinstein.com
"""
import os
import sys

import numpy as np
import pytest

import quantile_regression_rs as R

PY = os.environ.get("QR_PYTHON")
if not PY:
    pytest.skip("set QR_PYTHON to the Python port's directory", allow_module_level=True)
sys.path.insert(0, PY)
import ipqr as P_ip  # noqa: E402
import qr_extensions as P_ext  # noqa: E402
import rq as P_rq  # noqa: E402
import selection_qr as P_sel  # noqa: E402
import selection_qr_fast as P_f  # noqa: E402
import smoothed_qr as P_sm  # noqa: E402

rng = np.random.default_rng(20261124)
n = 500
X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
y = X @ np.array([1.0, 2.0, -1.0]) + rng.standard_t(3, size=n)
m = 800
Zs, Xs = rng.normal(size=m), rng.normal(size=m)
u = rng.normal(size=m)
e = 0.5 * u + np.sqrt(0.75) * rng.normal(size=m)
D = (0.3 + Xs + Zs + u > 0).astype(int)
Y = 1 + Xs + e


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_rq(tau):
    assert np.max(np.abs(R.rq(X, y, tau) - P_rq.rq(X, y, tau))) < 1e-10
    assert np.max(np.abs(R.rq(X, y, tau, method="cholesky") - P_f.rq_fast(X, y, tau))) < 1e-10


def test_rq_per_observation_tau():
    t = rng.uniform(0.2, 0.8, size=n)
    assert np.max(np.abs(R.rq(X, y, t) - P_rq.rq(X, y, t))) < 1e-10


def test_rqic():
    Rm, r = np.array([[0.0, 1.0, 0.0]]), np.array([2.5])
    assert np.max(np.abs(R.rqic(X, y, Rm, r, 0.5) - P_rq.rqic(X, y, Rm, r, 0.5))) < 1e-10


def test_ipqr_linear_and_nonlinear():
    f, df = (lambda x, b: x @ b), (lambda x, b: x)
    b = P_ip.ipqr(0.5, y, X, f, df, np.zeros(3))
    assert np.max(np.abs(R.ipqr(0.5, y, X, beta=np.zeros(3))["beta"] - b["beta"])) < 1e-10
    assert np.max(np.abs(R.ipqr(0.5, y, X, f, df, np.zeros(3))["beta"] - b["beta"])) < 1e-10
    xx = rng.uniform(0, 3, n)
    yy = 2 * np.exp(0.5 * xx) + rng.normal(size=n)
    fn = lambda x, b: b[0] * np.exp(b[1] * x)  # noqa: E731
    dfn = lambda x, b: np.column_stack([np.exp(b[1] * x), b[0] * x * np.exp(b[1] * x)])  # noqa: E731
    a = R.ipqr(0.3, yy, xx, fn, dfn, np.array([1.0, 0.3]))
    b = P_ip.ipqr(0.3, yy, xx, fn, dfn, np.array([1.0, 0.3]))
    assert np.max(np.abs(a["beta"] - b["beta"])) < 1e-8


def test_ipqr_python_error_propagates():
    with pytest.raises(ZeroDivisionError):
        R.ipqr(0.5, y, X, lambda x, b: 1 / 0, lambda x, b: x, np.zeros(3))


def test_selection_qr():
    a, b = R.selection_qr(D, Y, Xs, Zs), P_sel.selection_qr(D, Y, Xs, Zs)
    assert a["rho"] == b["rho"]
    assert np.max(np.abs(a["beta"] - b["beta"])) < 1e-10
    assert np.allclose(a["objective"], b["objective"], rtol=1e-8, atol=1e-300)
    assert np.max(np.abs(a["pZ"] - b["pZ"])) < 1e-12


def test_selection_qr_weighted():
    w = rng.exponential(size=m)
    a, b = R.selection_qr_weighted(D, Y, Xs, Zs, w=w), P_ext.selection_qr_weighted(D, Y, Xs, Zs, w=w)
    assert a["rho"] == b["rho"]
    assert np.max(np.abs(a["beta"] - b["beta"])) < 1e-10


def test_selection_qr_fast():
    a = R.selection_qr_fast(D, Y, Xs, Zs, preprocess=True, search="coarse")
    b = P_f.selection_qr_fast(D, Y, Xs, Zs, preprocess=True, search="coarse")
    assert a["rho"] == b["rho"]
    assert np.max(np.abs(a["beta"] - b["beta"])) < 1e-8


def test_helpers():
    Zc = np.column_stack([Xs, Zs])
    assert np.max(np.abs(R.probit(D, Zc) - P_sel.probit(D, Zc))) < 1e-10
    assert abs(R.bivariate_normal_cdf(0.3, -0.2, 0.6) - P_sel.bivariate_normal_cdf(0.3, -0.2, 0.6)) < 1e-14
    p = rng.uniform(0.1, 0.9, 50)
    assert np.max(np.abs(R.rotated_tau(0.3, p, 0.4) - P_sel.rotated_tau(0.3, p, 0.4))) < 1e-12
    for meth in ("iid", "ker"):
        assert np.max(np.abs(R.rq_se(X, y, 0.5, method=meth)[0] - P_ext.rq_se(X, y, 0.5, method=meth)[0])) < 1e-10
    beta = np.sort(rng.normal(size=(3, 9)), axis=1)
    assert np.max(np.abs(R.rearrange(beta, X[:10]) - P_ext.rearrange(beta, X[:10]))) < 1e-12
    assert np.max(np.abs(R.conquer(X, y, 0.5) - P_sm.conquer(X, y, 0.5))) < 1e-8


def test_bootstrap_shapes_and_threads():
    bt = R.selection_qr_bootstrap(D, Y, Xs, Zs, reps=10, seed=3)
    assert bt["beta_draws"].shape == (10, 2, 19) and bt["beta_se"].shape == (2, 19)
    again = R.selection_qr_bootstrap(D, Y, Xs, Zs, reps=10, seed=3)
    assert np.array_equal(bt["beta_draws"], again["beta_draws"])
    cb = R.conquer_bootstrap(X, y, 0.5, reps=30, seed=1)
    assert cb["draws"].shape == (30, 3) and cb["percentile"].shape == (3, 2)


def test_bad_input():
    with pytest.raises(ValueError):
        R.rq(X, y[:5])
