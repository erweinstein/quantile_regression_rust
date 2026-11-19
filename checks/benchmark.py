'''benchmark.py -- compute-only timings of the Python port (erweinstein/quantile_regression) and this Rust
translation on the same data sets, one thread (and the Rust selection estimator on 2 threads too).
Usage: python checks/benchmark.py <path to erweinstein/quantile_regression>   (after cargo build --release --examples)
Writes checks/benchmark.csv.

Eliot R. Weinstein / eliotw@uchicago.edu / erweinstein.com
'''
import csv, os, subprocess, sys, tempfile, time, warnings
import numpy as np
PY = os.path.abspath(sys.argv[1]); sys.path.insert(0, PY)
from rq import rq  # noqa: E402
from selection_qr import selection_qr  # noqa: E402
from selection_qr_fast import rq_fast, selection_qr_fast  # noqa: E402
from ipqr import ipqr  # noqa: E402
warnings.simplefilter('ignore')
# numpy's BLAS is limited to one thread by the environment variables set before Python starts (see the README)
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, '..', 'target', 'release', 'examples', 'bench')
TMP = tempfile.mkdtemp()
REPS = 3


def rust(kind, path, threads=1, tau=0.5):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    out = [subprocess.run([BENCH, kind, path, str(tau)], capture_output=True, text=True, env=env, check=True).stdout
           for _ in range(REPS)]
    return np.median([float(o.split(',')[0]) for o in out]), out[0].strip().split(',', 1)[1]


def py(f):
    ts, r = [], None
    for _ in range(REPS):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), r


rows = []
# 1. rq at tau = .5, p = 3
for n in (1000, 10000, 100000):
    rng = np.random.default_rng(n)
    x = rng.normal(size=(n, 2)); y = 1 + x @ [1, -0.5] + rng.standard_t(3, size=n)
    X = np.column_stack([np.ones(n), x])
    path = os.path.join(TMP, f'rq{n}.csv'); np.savetxt(path, np.column_stack([y, x]), delimiter=',', header='y,x1,x2', comments='')
    for name, f, kind in (('rq (QR / lstsq)', lambda: rq(X, y, 0.5), 'rq'), ('rq_fast (Cholesky)', lambda: rq_fast(X, y, 0.5), 'rq_chol')):
        tp, _ = py(f); tr, _ = rust(kind, path)
        rows.append((name, n, 1, tp, tr)); print(rows[-1], flush=True)
# 2. the selection-corrected estimator (Code_QRS.m; and the fast version with preprocessing)
for n in (2000, 8000):
    rng = np.random.default_rng(9000)
    X = rng.normal(size=n); B = rng.normal(size=n)
    u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n).T
    D = (0.3 + 0.5 * X + B + v > 0).astype(int); Y = 1 + X + (0.5 + 0.3 * X) * u
    path = os.path.join(TMP, f'sel{n}.csv'); np.savetxt(path, np.column_stack([D, Y, X, B]), delimiter=',', header='D,Y,X,B', comments='')
    tp, rp = py(lambda: selection_qr(D, Y, X, B))
    for th in (1, 2):
        tr, rr = rust('sel', path, th)
        rows.append(('selection_qr (Code_QRS.m)', n, th, tp, tr)); print(rows[-1], rp['rho'], rr, flush=True)
    tp, rp = py(lambda: selection_qr_fast(D, Y, X, B, preprocess=True))
    tr, rr = rust('sel_fast', path, 1)
    rows.append(('selection_qr_fast, preprocessing', n, 1, tp, tr)); print(rows[-1], rp['rho'], rr, flush=True)
# 3. ipqr, exponential model (tests/test_rq.py's design)
rng = np.random.default_rng(0); x = rng.uniform(0, 2, 5000); y = 2 * np.exp(0.7 * x) + rng.normal(scale=0.5, size=5000)
path = os.path.join(TMP, 'ipqr.csv'); np.savetxt(path, np.column_stack([x, y]), delimiter=',', header='x,y', comments='')
f = lambda x, b: b[0] * np.exp(b[1] * x)                                                    # noqa: E731
df = lambda x, b: np.column_stack([np.exp(b[1] * x), b[0] * x * np.exp(b[1] * x)])          # noqa: E731
tp, _ = py(lambda: ipqr(0.5, y, x, f, df, np.array([1.0, 0.3]))); tr, _ = rust('ipqr', path)
rows.append(('ipqr, exponential model', 5000, 1, tp, tr)); print(rows[-1], flush=True)

with open(os.path.join(HERE, 'benchmark.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['task', 'n', 'rust_threads', 'python_seconds', 'rust_seconds', 'ratio'])
    for r in rows:
        w.writerow([r[0], r[1], r[2], f'{r[3]:.4f}', f'{r[4]:.4f}', f'{r[3] / r[4]:.1f}'])
print(open(os.path.join(HERE, 'benchmark.csv')).read())
