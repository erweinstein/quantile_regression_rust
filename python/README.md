# quantile_regression_rs: Python bindings

Python bindings, through [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs), for the `quantile_regression`
crate in the folder above. The functions mirror my Python port,
[erweinstein/quantile_regression](https://github.com/erweinstein/quantile_regression): the same names, argument
order, defaults and returned dicts, taking and returning NumPy arrays.

Eliot R. Weinstein · eliotw@uchicago.edu · erweinstein.com

## Install

Rust 1.97 (cargo) and Python 3.9 or later. From this folder:

```
pip install maturin
maturin develop --release              # into the active virtual environment
# or
maturin build --release && pip install target/wheels/quantile_regression_rs-*.whl
```

The wheel uses Python's stable ABI (abi3), so one build serves Python 3.9 and later on the same platform.

## Use

```python
import numpy as np
import quantile_regression_rs as qr

X = np.column_stack([np.ones(n), x1, x2])        # rq does not add the constant, as in rq.m
b = qr.rq(X, y, 0.5)                              # the median regression
b = qr.rq(X, y, tau_i)                            # one tau per observation
bc = qr.rqic(X, y, np.array([[0, 0, 1.0]]), np.array([0.5]), 0.5)   # with b3 >= 0.5

fit = qr.ipqr(0.5, y, x, lambda x, b: b[0] * np.exp(b[1] * x),
              lambda x, b: np.column_stack([np.exp(b[1] * x), b[0] * x * np.exp(b[1] * x)]),
              np.array([1.0, 0.3]))               # nonlinear: Python functions called from Rust
fit = qr.ipqr(0.5, y, X, beta=np.zeros(3))        # linear: entirely in Rust

f = qr.selection_qr(D, Y, X, B)                   # Arellano and Bonhomme (2017), as Code_QRS.m
f["rho"], f["beta"]                               # beta: (1 + dim X) x 19, columns tau = .05, ..., .95
bs = qr.selection_qr_bootstrap(D, Y, X, B, reps=200, seed=2017)
ff = qr.selection_qr_fast(D, Y, X, B, preprocess=True, search="coarse")
```

Also: `rq_objective`, `probit`, `bivariate_normal_cdf`, `rotated_tau`, `selection_qr_weighted`, `rq_se`,
`hall_sheather`, `rearrange`, `conquer`, `conquer_bootstrap` and `default_bandwidth`. Each has a docstring
(`help(qr.selection_qr)`).

## Differences from the Python port

- **The bootstraps' random draws.** `selection_qr_bootstrap` and `conquer_bootstrap` take an integer seed and
  give draw r its own Xoshiro256++ stream, as the Rust crate does; the Python port uses one NumPy generator. The
  draws therefore differ, and so do the bootstrap standard errors, by Monte Carlo error. The draws run in
  parallel (`RAYON_NUM_THREADS` sets the threads) and do not depend on the number of threads.
- **`rq`'s solver options** are keyword arguments (`beta`, `small`, `max_it`, `method="qr"` or `"cholesky"`), where
  the port passes `**kwargs` to `lp_fnm`. `method="cholesky"` is the port's `rq_fast`.
- **`ipqr`'s `func` and `dfunc` may be omitted** for the linear model, which then runs without calling back into
  Python. With them, every evaluation crosses from Rust to Python, so a nonlinear `ipqr` is no faster than the
  port; an exception raised in them is passed through.
- **The GIL is released** during the long computations, so they can run in Python threads.

## Speed from Python

One thread on each side (`RAYON_NUM_THREADS=1`, BLAS limited to one thread), on an idle 2-core Linux container,
medians of 3–7 calls (`tests/timing.py`):

| Task | Python port | Bindings | Ratio |
|---|---|---|---|
| `rq`, n = 2,000, p = 5 | 7.6 ms | 1.8 ms | 4.2× |
| `conquer`, n = 2,000, p = 5 | 2.3 ms | 1.0 ms | 2.3× |
| `selection_qr`, n = 2,000 | 2.73 s | 0.42 s | 6.5× |
| `selection_qr_fast`, preprocessing and coarse search, n = 2,000 | 0.64 s | 0.074 s | 8.6× |

With more threads the selection estimator and the bootstraps gain further (see the main README). The cost of
crossing from Python to Rust, copying the arrays in and out, is small next to these times.

## Checks

`tests/test_against_python.py` compares every function with the Python port on the same data (13 tests):

```
QR_PYTHON=/path/to/quantile_regression python -m pytest tests
```

The point estimates agree to rounding: `rq`, `rqic`, `probit`, `selection_qr` (ρ identical, β within 1e-15),
`rq_se` and `rotated_tau` within 1e-12 or better; `ipqr` within 2e-13 (linear) and to every printed digit
(exponential model); `conquer` within 1e-11; `selection_qr_fast` with preprocessing and the coarse search gives
the same ρ, with β within 1e-9.

## License

As for the crate: the Apache License, Version 2.0 or the MIT license, at your option, with the third-party
notices in `NOTICE.md`. The copies of `LICENSE-APACHE`, `LICENSE-MIT` and `NOTICE.md` in this folder are there so
that the wheel carries them; the ones in the folder above are the originals.
