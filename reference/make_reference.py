'''make_reference.py -- reference outputs from the Python code in erweinstein/quantile_regression, for the
Rust tests (tests/reference/*.json). The Python repository is used as it is; nothing in it is changed.

Usage:  python reference/make_reference.py <path to a clone of erweinstein/quantile_regression>

Eliot R. Weinstein
eliotw@uchicago.edu
erweinstein.com
'''
import json
import os
import subprocess
import sys
import warnings

import numpy as np

PY = os.path.abspath(sys.argv[1])
sys.path.insert(0, PY)
from rq import rq, rqic, rq_objective  # noqa: E402
from ipqr import ipqr  # noqa: E402
from selection_qr import bivariate_normal_cdf, probit, selection_qr  # noqa: E402
from selection_qr_fast import rq_fast, rq_preprocessed, selection_qr_fast, check_objective  # noqa: E402
from qr_extensions import rq_se, weighted_rq, selection_qr_weighted, rearrange  # noqa: E402
from smoothed_qr import conquer, default_bandwidth  # noqa: E402
from scipy.stats import norm  # noqa: E402

warnings.simplefilter('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'reference')
os.makedirs(OUT, exist_ok=True)


def L(a):
    return np.asarray(a, dtype=float).tolist()


def save(name, obj):
    commit = subprocess.run(['git', '-C', PY, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    obj = {'python_commit': commit, **obj}
    with open(os.path.join(OUT, name), 'w') as fh:
        json.dump(obj, fh)
    print('wrote', name, flush=True)


# ---- data designs (as in the Python tests and checks) ------------------------------------------------------
def data_rq(n=300, seed=0):                       # tests/test_rq.py
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(size=n)])
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.standard_t(3, size=n) * (1 + X[:, 2])
    return X, y


def problem(n, seed=0):                           # tests/test_fast_and_smoothed.py
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    X = np.column_stack([np.ones(n), x])
    y = 1 + x @ [1, -0.5] + (1 + 0.5 * np.abs(x[:, 0])) * rng.standard_t(4, size=n)
    tau = rng.uniform(0.2, 0.8, size=n)
    return X, y, tau


def data_sel(n, seed):                            # checks/timing.py
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n); B = rng.normal(size=n)
    u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n).T
    D = (0.3 + 0.5 * X + B + v > 0).astype(int)
    return D, 1 + X + (0.5 + 0.3 * X) * u, X, B


TAUS = [0.1, 0.25, 0.5, 0.75, 0.9]

# ---- stage 1: rq, rqic, objective ------------------------------------------------------------------------
cases = []
for (n, seed) in [(300, 0), (1000, 1)]:
    X, y = data_rq(n, seed)
    fits = [{'tau': t, 'coef': L(rq(X, y, t)), 'coef_fast': L(rq_fast(X, y, t)),
             'objective': rq_objective(X, y, rq(X, y, t), t)} for t in TAUS]
    cases.append({'X': L(X), 'y': L(y), 'fits': fits})
X, y, tau = problem(800)
obs = {'X': L(X), 'y': L(y), 'tau': L(tau), 'coef': L(rq(X, y, tau)), 'coef_fast': L(rq_fast(X, y, tau)),
       'objective': check_objective(X, y, tau, rq(X, y, tau))}
X, y = data_rq()
rqic_cases = [{'R': [[0, 0, 1]], 'r': [-10.0], 'coef': L(rqic(X, y, [[0, 0, 1]], [-10.0], 0.5))},
              {'R': [[0, 0, 1]], 'r': [0.5], 'coef': L(rqic(X, y, [[0, 0, 1]], [0.5], 0.5))},
              {'R': [[0, 1, 0], [0, 0, 1]], 'r': [2.5, -2.0], 'coef': L(rqic(X, y, [[0, 1, 0], [0, 0, 1]], [2.5, -2.0], 0.5))}]
save('rq.json', {'cases': cases, 'per_obs': obs, 'rqic': {'X': L(X), 'y': L(y), 'tau': 0.5, 'cases': rqic_cases}})

# ---- normal and bivariate normal functions ---------------------------------------------------------------
xs = np.linspace(-9, 9, 181)
us = np.r_[1e-12, 1e-8, 1e-4, np.linspace(0.001, 0.999, 999), 1 - 1e-4, 1 - 1e-8]
rng = np.random.default_rng(7)
H, K = rng.normal(scale=1.5, size=400), rng.normal(scale=1.5, size=400)
H[:5], K[5:10] = 0.0, 0.0
bvn = []
for rho in [-0.999, -0.98, -0.9, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 0.9, 0.98, 0.999]:
    bvn.append({'rho': rho, 'values': L(bivariate_normal_cdf(H, K, rho))})
save('normal.json', {'x': L(xs), 'cdf': L(norm.cdf(xs)), 'pdf': L(norm.pdf(xs)), 'u': L(us), 'ppf': L(norm.ppf(us)),
                     'h': L(H), 'k': L(K), 'bvn': bvn})

# ---- stage 2: probit and selection-corrected QR ----------------------------------------------------------
sel = []
for (n, seed) in [(500, 1), (500, 2), (2000, 9000), (2000, 9001)]:
    D, Y, X, B = data_sel(n, seed)
    r = selection_qr(D, Y, X, B)
    sel.append({'n': n, 'seed': seed, 'D': L(D), 'Y': L(Y), 'X': L(X), 'B': L(B),
                'gamma': L(r['gamma']), 'rho': r['rho'], 'beta': L(r['beta']), 'objective': L(r['objective']),
                'vecrho': L(r['vecrho']), 'taus': L(r['taus'])})
save('selection.json', {'cases': sel})

# weighted estimator with fixed weights, and weighted_rq
D, Y, X, B = data_sel(500, 1)
w = np.random.default_rng(11).exponential(size=500)
rw = selection_qr_weighted(D, Y, X, B, w=w)
Xr, yr = data_rq()
wr = np.random.default_rng(12).exponential(size=300)
save('weighted.json', {'D': L(D), 'Y': L(Y), 'X': L(X), 'B': L(B), 'w': L(w), 'gamma': L(rw['gamma']), 'rho': rw['rho'],
                       'beta': L(rw['beta']), 'objective': L(rw['objective']),
                       'rq': {'X': L(Xr), 'y': L(yr), 'w': L(wr), 'tau': 0.3, 'coef': L(weighted_rq(Xr, yr, 0.3, wr))}})

# ---- stage 3: ipqr -----------------------------------------------------------------------------------------
X, y = data_rq()
lin = [{'tau': t, **{k: (L(v) if k == 'beta' else v) for k, v in
                      ipqr(t, y, X, lambda x, b: x @ b, lambda x, b: x, np.zeros(3)).items()}} for t in (0.25, 0.5, 0.75)]
rng = np.random.default_rng(0)
x = rng.uniform(0, 2, 500)
ye = 2 * np.exp(0.7 * x) + rng.normal(scale=0.5, size=500)
f = lambda x, b: b[0] * np.exp(b[1] * x)                                                       # noqa: E731
df = lambda x, b: np.column_stack([np.exp(b[1] * x), b[0] * x * np.exp(b[1] * x)])             # noqa: E731
ex = [{'tau': t, **{k: (L(v) if k == 'beta' else v) for k, v in ipqr(t, ye, x, f, df, np.array([1.0, 0.3])).items()}}
      for t in (0.25, 0.5, 0.75)]
save('ipqr.json', {'X': L(X), 'y': L(y), 'linear': lin, 'x_exp': L(x), 'y_exp': L(ye), 'exponential': ex})

# ---- stage 4: extensions -------------------------------------------------------------------------------
X, y = data_rq(1000, 1)
se = [{'tau': t, 'ker': L(rq_se(X, y, t, method='ker')[0]), 'iid': L(rq_se(X, y, t, method='iid')[0])} for t in TAUS]
Xp, yp, taup = problem(5000, 3)
b0 = rq_fast(Xp, yp, np.clip(taup + 0.01, 0.01, 0.99))
bp, info = rq_preprocessed(Xp, yp, taup, b0)
D, Y, Xs, Bs = data_sel(2000, 9000)
fast = {}
for name, kw in {'exact': dict(preprocess=False), 'pre': dict(preprocess=True), 'pre_coarse': dict(preprocess=True, search='coarse'),
                 'paper_pre_coarse': dict(settings='paper', preprocess=True, search='coarse'), 'tie': dict(tie_tol=1e-7)}.items():
    r = selection_qr_fast(D, Y, Xs, Bs, **kw)
    fast[name] = {'rho': r['rho'], 'beta': L(r['beta']), 'objective': L(np.nan_to_num(r['objective'], nan=-1.0))}
Xc, yc, _ = problem(2000, 5)
cq = [{'tau': t, 'coef': L(conquer(Xc, yc, t)), 'h': default_bandwidth(*Xc.shape)} for t in (0.25, 0.5, 0.75)]
beta_cross = np.array([[0.0, 0.1, 0.05, 0.3], [1.0, 0.8, 1.2, 0.9]])
save('extensions.json', {'se': {'X': L(X), 'y': L(y), 'cases': se},
                         'preprocessed': {'X': L(Xp), 'y': L(yp), 'tau': L(taup), 'b0': L(b0), 'coef': L(bp),
                                          'objective': check_objective(Xp, yp, taup, bp),
                                          'objective_full': check_objective(Xp, yp, taup, rq_fast(Xp, yp, taup))},
                         'fast': {'D': L(D), 'Y': L(Y), 'X': L(Xs), 'B': L(Bs), **fast},
                         'conquer': {'X': L(Xc), 'y': L(yc), 'cases': cq},
                         'rearrange': {'beta': L(beta_cross), 'x': [[1, -2.0], [1, 0.5]],
                                       'curves': L(rearrange(beta_cross, [[1, -2.0], [1, 0.5]]))}})
