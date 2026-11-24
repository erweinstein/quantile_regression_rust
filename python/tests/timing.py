"""Times the Python bindings against the Python port on the same data (one thread each).

    QR_PYTHON=/path/to/quantile_regression RAYON_NUM_THREADS=1 python python/tests/timing.py

Prints one CSV row per task: the task, the port's median seconds, the bindings' median seconds, and the ratio.

Eliot R. Weinstein
eliotw@uchicago.edu
erweinstein.com
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import quantile_regression_rs as R  # noqa: E402

sys.path.insert(0, os.environ["QR_PYTHON"])
import rq as P_rq  # noqa: E402
import selection_qr as P_sel  # noqa: E402
import selection_qr_fast as P_f  # noqa: E402
import smoothed_qr as P_sm  # noqa: E402


def med(f, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


rng = np.random.default_rng(1)
n = 2000
X = np.column_stack([np.ones(n), rng.normal(size=(n, 4))])
y = X @ np.array([1.0, 1, -1, 0.5, 0]) + rng.standard_t(3, size=n)
m = 2000
Zs, Xs, u = rng.normal(size=m), rng.normal(size=m), rng.normal(size=m)
D = (0.3 + Xs + Zs + u > 0).astype(int)
Y = 1 + Xs + 0.5 * u + np.sqrt(0.75) * rng.normal(size=m)

tasks = [
    ("rq, n = 2000, p = 5", lambda: P_rq.rq(X, y, 0.5), lambda: R.rq(X, y, 0.5), 7),
    ("conquer, n = 2000, p = 5", lambda: P_sm.conquer(X, y, 0.5), lambda: R.conquer(X, y, 0.5), 7),
    ("selection_qr, n = 2000", lambda: P_sel.selection_qr(D, Y, Xs, Zs), lambda: R.selection_qr(D, Y, Xs, Zs), 3),
    ("selection_qr_fast, preprocess + coarse", lambda: P_f.selection_qr_fast(D, Y, Xs, Zs, preprocess=True, search="coarse"),
     lambda: R.selection_qr_fast(D, Y, Xs, Zs, preprocess=True, search="coarse"), 3),
]
print("task,python_port_secs,bindings_secs,ratio")
for name, fp, fr, reps in tasks:
    a, b = med(fp, reps), med(fr, reps)
    print(f"{name},{a:.4f},{b:.4f},{a / b:.1f}", flush=True)
