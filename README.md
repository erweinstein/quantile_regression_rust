# quantile_regression_rust

A Rust translation of the interior point quantile regression code of Koenker, Morillo and Eilers (`rq.m`,
`rqic.m`) and Koenker and Park (`ipqr.m`), and Rust replication code for Arellano and Bonhomme (2017).

Eliot R. Weinstein · eliotw@uchicago.edu · erweinstein.com

I translated this from my Python port, [erweinstein/quantile_regression](https://github.com/erweinstein/quantile_regression),
which stays as it is. That port follows the MATLAB originals line by line, and this crate follows the port
statement by statement, with the MATLAB lines in the comments where the correspondence is not obvious. Every
function is checked against the Python code's output on the same data (below). The algorithms and the MATLAB
code are their authors'; see `NOTICE.md` for credit and for the scope of the license.

## What is here

| Rust | Translates | What it does |
|---|---|---|
| `rq::rq`, `rq::lp_fnm`, `rq::bound` | `rq.m` (`rq.py`) | Regression quantiles by the Frisch–Newton interior point method (Mehrotra predictor-corrector steps), with one τ or one τ per observation |
| `rq::rqic`, `rq::lp_fnm_ineq` | `rqic.m` | The same with linear inequality constraints R b ≥ r |
| `ipqr::ipqr` | `ipqr.m` (`ipqr.py`) | Nonlinear quantile regression by Koenker and Park's interior point algorithm; the model is any type implementing `ipqr::Model` |
| `selection::selection_qr` | Arellano and Bonhomme's `Code_QRS.m` (`selection_qr.py`) | Selection-corrected quantile regression: probit, Gaussian copula, grid search for ρ, rotated quantile regressions |
| `selection::selection_qr_weighted`, `selection_qr_bootstrap` | `qr_extensions.py` | The estimator with observation weights, and its weighted (multiplier) bootstrap with Exp(1) weights |
| `fast::selection_qr_fast`, `fast::rq_preprocessed` | `selection_qr_fast.py` | Warm starts along the ρ grid, Portnoy–Koenker preprocessing for observation-specific τ, a coarse-then-fine search, the paper's search settings, and a tie tolerance |
| `rq::Method::Cholesky` (`rq::rq_fast`) | `lp_fnm_fast` in `selection_qr_fast.py` | The Newton systems by Cholesky instead of QR (the same iterates up to rounding) |
| `extensions::rq_se`, `extensions::rearrange` | `qr_extensions.py` | Standard errors (Powell's kernel sandwich, or Koenker–Bassett with a difference-quotient sparsity), and Chernozhukov–Fernández-Val–Galichon rearrangement |
| `smoothed::conquer`, `conquer_bootstrap` | `smoothed_qr.py` | Convolution-smoothed quantile regression and its multiplier bootstrap |
| `normal` | `scipy.stats.norm`, `scipy.special.owens_t` | Φ, φ, Φ⁻¹, and the bivariate normal cdf (Genz 2004) that replaces MATLAB's `copulacdf` |
| `src/bin/qrs.rs` | how `Code_QRS.m` is run | The Arellano–Bonhomme estimator on a CSV file, from the command line |

"Replication code" means a translation of the authors' estimator, `Code_QRS.m`: given data, it computes what
their MATLAB code computes. Their empirical data are not included.

## Checks against the Python port

The reference outputs in `tests/reference/` were written by `reference/make_reference.py`, which runs the
Python code (commit e86f388 of erweinstein/quantile_regression) on simulated data sets taken from its tests and
checks. `cargo test --release` compares the Rust results with them:

| Function | Agreement with the Python port |
|---|---|
| `rq`, `rq_fast` (2 data sets × 5 τ, and τ per observation) | coefficients within 4.4 × 10⁻¹⁵; objectives within 1.5 × 10⁻¹⁵ relative |
| `rqic` (slack, binding, two constraints) | within 8.9 × 10⁻¹⁶ |
| Φ, Φ⁻¹ | within 3.1 × 10⁻¹⁵ and 1.8 × 10⁻¹⁵ (relative and absolute) of scipy |
| bivariate normal cdf (400 points × 13 values of ρ) | within 4 × 10⁻¹³, which is the Python version's own shift of h = 0 and k = 0 by 10⁻¹² |
| probit | within 2.2 × 10⁻¹⁶ |
| `selection_qr` (4 data sets, n = 500 and 2,000) | the same ρ on all 4; β within 8.9 × 10⁻¹⁶ |
| `selection_qr_weighted` (fixed weights) | the same ρ; β within 1.1 × 10⁻¹⁵ |
| `ipqr` (linear model at 3 τ; exponential model at 3 τ) | the same number of iterations in all 6; β within 9.2 × 10⁻¹³ |
| `rq_se` (kernel and iid, 5 τ) | within 3.9 × 10⁻¹⁵ relative |
| `rq_preprocessed` (n = 5,000) | the full problem's optimum (objective within 6 × 10⁻¹² relative); coefficients within 1.8 × 10⁻¹⁰ of Python's |
| `selection_qr_fast` (exact, preprocessed, preprocessed and coarse, paper settings, tie tolerance) | the same ρ in all 5, and the same grid points evaluated; β within 2.7 × 10⁻¹¹ |
| `conquer` (3 τ) | within 1.4 × 10⁻⁸, its own stopping tolerance on the Newton step |
| `rearrange` | exact |

The weighted bootstrap cannot match draw for draw, because it uses its own generator (below). On the
n = 2,000 data set with 50 draws each (`checks/bootstrap_check.py`), the standard errors agree within the Monte
Carlo error of 50 draws, about ±10%: 0.074 against Python's 0.080 for ρ, and (0.023, 0.018) against
(0.025, 0.017) for β(0.5). A test checks that the bootstrap gives the same draws on 1 and 2 threads.

## Speed

Compute time only, medians of 3 runs, on an idle 2-core Linux container (`checks/benchmark.py`,
`examples/bench.rs`, results in `checks/benchmark.csv`). Python 3.13 with NumPy 2.5 and SciPy 1.16 (BLAS limited
to one thread); Rust 1.97 with opt-level 3 and link-time optimization.

| Task | n | Python | Rust, 1 thread | Rust, 2 threads |
|---|---|---|---|---|
| `rq`, τ = 0.5, p = 3 | 1,000 | 2.5 ms | 0.6 ms (4.3×) | |
| | 10,000 | 21.5 ms | 12.1 ms (1.8×) | |
| | 100,000 | 287 ms | 113 ms (2.6×) | |
| `rq_fast` (Cholesky) | 100,000 | 195 ms | 106 ms (1.8×) | |
| `selection_qr` (`Code_QRS.m`) | 2,000 | 2.00 s | 0.37 s (5.4×) | 0.20 s (10.1×) |
| | 8,000 | 7.09 s | 1.79 s (4.0×) | 0.96 s (7.4×) |
| `selection_qr_fast`, preprocessing | 2,000 | 1.12 s | 0.12 s (9.3×) | |
| | 8,000 | 1.99 s | 0.42 s (4.7×) | |
| `ipqr`, exponential model | 5,000 | 9.3 ms | 7.2 ms (1.3×) | |
| weighted bootstrap, 50 draws | 2,000 | 81 s | | 14 s (5.8×; whole program) |

- The gain on `rq` alone is modest because NumPy already runs each vector operation in C. What Rust removes is
  the dozens of separate passes over length-n vectors in each interior point iteration, so the gain is largest
  for small n.
- The selection estimator gains more. The copula grid, the final quantiles and the bootstrap draws are
  independent fits, and they run in parallel with rayon. Each fit is deterministic, so the results do not depend
  on the number of threads (`RAYON_NUM_THREADS` sets it). The rotated quantile levels reuse Φ⁻¹(p(Z)) and the
  parts of Genz's bivariate normal algorithm that depend only on ρ across the 415 evaluations.
- `ipqr` gains little: its time is mostly the line search's evaluations of the model, `exp` evaluated at all n
  points, which costs about the same in either language.

## Using it

As a dependency in `Cargo.toml`:

```toml
[dependencies]
quantile_regression = { git = "https://github.com/erweinstein/quantile_regression_rust" }
```

```rust
use quantile_regression::linalg::Matrix;
use quantile_regression::rq::{rq, rqic};
use quantile_regression::selection::{selection_qr, selection_qr_bootstrap, SelectionOptions};

// X includes the column of ones (rq does not add one, as in rq.m)
let x = Matrix::from_columns(&[&vec![1.0; n], &x1, &x2]);
let b = rq(&x, &y, 0.5)?;                                   // the median regression
let bc = rqic(&x, &y, &Matrix::from_row_major(1, 3, vec![0.0, 0.0, 1.0]), &[0.5], 0.5)?;  // b3 >= 0.5

// Code_QRS.m: d participation, y outcome, x covariates and b excluded covariates (no constants)
let fit = selection_qr(&d, &y, &xm, &bm, &SelectionOptions::default())?;
println!("rho = {}, beta(0.5) = {:?}", fit.rho, (0..fit.beta.nrow).map(|k| fit.beta.get(k, 9)).collect::<Vec<_>>());
let bs = selection_qr_bootstrap(&d, &y, &xm, &bm, 200, 2017, &SelectionOptions::default())?;
```

From the command line:

```
cargo build --release
./target/release/qrs data.csv --d D --y Y --x X1,X2 --b B1 [--bootstrap 200 --seed 2017] [--fast | --paper] [--out beta.csv]
```

It writes β(τ) for τ = 0.05, …, 0.95 (with `--bootstrap`, also weighted bootstrap standard errors) and prints ρ
and the probit coefficients.

## Differences from the Python port

- **Least squares.** MATLAB's `A' \ c` and the Python port's `numpy.linalg.lstsq` (SVD) become Householder QR,
  which is what MATLAB's backslash uses for a rectangular system. The rank check applies numpy's tolerance
  formula to the diagonal of R instead of to singular values.
- **The bivariate normal cdf** is Genz's algorithm instead of the Owen's T formula; the two agree to about
  10⁻¹⁵ away from h = 0 and k = 0.
- **Φ** uses libm's `erfc`. I tried statrs's first, and it was accurate only to about 10⁻¹⁰, which moved the probit
  coefficients by about 3 × 10⁻¹¹.
- **Bootstrap draws** use one Xoshiro256++ stream per draw, seeded from (seed, draw), instead of numpy's
  generator. So the draws differ from Python's, but they do not depend on the number of threads.
- **Non-convergence** is reported in the result (`rq_fit`, `RqFit::converged`) instead of as a warning.

## Possible problems

- **Ties at the fitted line.** At the solution, p observations lie exactly on each rotated regression line, and
  the interior point method leaves them at about ±10⁻⁹. `Code_QRS.m`'s moment 1{Y ≤ X′b} counts them by that
  rounding. Any change in the arithmetic can move them across, and with them ρ by a grid step or two (0.02–0.04).
  This includes preprocessing, a different least-squares routine, or a different machine. On all the reference
  data sets the Rust and Python ρ agree, but that is not guaranteed for every data set. `tie_tol` in
  `selection_qr_fast` makes the count independent of the rounding.
- **The coarse search** is exact only when the full-grid minimizer lies near one of the best coarse points, which
  it reports it cannot guarantee (as in the Python port).
- **The benchmarks are small**: a 2-core machine and a 4-vCPU GitHub runner (below), a few data sets. Beyond 4
  threads the scaling is simulated, not measured.
- **No Python bindings yet.** To call this from Python, a PyO3 wrapper would be the next step.

## More threads and more cores

`examples/multicore.rs` runs `selection_qr` (n = 8,000), `selection_qr_fast` with preprocessing, the weighted
bootstrap (16 draws, n = 2,000) and the conquer bootstrap (400 draws), and prints a digest of every bit of each
result. `checks/multicore.sh` runs it at 1, 2, 3, 4, 6 and 8 threads, and `.github/workflows/multicore.yml` runs
that on a GitHub-hosted runner whenever the `multicore-check` branch is pushed
(`checks/multicore_github_runner.txt`).

- **The same bits everywhere.** The digests are identical at every thread count (1–16 tried), on three
  processors (an Intel Xeon here; AMD EPYC 7763 and EPYC 9V74 on two runner jobs), and with two compilers
  (rustc 1.97 and 1.100).
- **Scaling, measured.** The runner's 4 vCPUs are 2 physical cores with 2 hyperthreads each. On it, 2 threads
  give 1.9–2.0× (`selection_qr` 2.486 → 1.289 s; the bootstrap 8.82 → 4.43 s) and 4 threads 2.7–2.9×, that is,
  the second hyperthread on each core adds about 40%. `selection_qr_fast` with preprocessing does not speed
  up, because its warm-started sweep along the ρ grid is sequential by design.
- **Scaling, simulated.** Timing each independent piece of `selection_qr` on one thread
  (`multicore --tasks`) and scheduling the pieces on k workers as rayon does predicts the 2-thread time to
  within 1% (1.224 s predicted, 1.216 s measured on the runner). On k physical cores it predicts about 3.9× on 4,
  7.4× on 8, 13× on 16 and 35× on 64 for one estimate: there are only 99 grid points and 19 quantiles to share
  out. A bootstrap of 200 draws scales further (8× on 8, 15× on 16, 50× on 64). The simulation ignores what cores
  share (memory bandwidth, caches, clock speed), so these are upper bounds.

## Reproducing the checks

```
cargo test --release                                              # all checks against the reference outputs
python reference/make_reference.py <path to quantile_regression>  # regenerate the reference outputs
cargo build --release --examples
OPENBLAS_NUM_THREADS=1 python checks/benchmark.py <path to quantile_regression>
bash checks/multicore.sh                                          # digests at several thread counts
```

## References

- Arellano, M. and S. Bonhomme (2017). Quantile selection models with an application to understanding changes in
  wage inequality. *Econometrica* 85(1), 1–28.
- Chernozhukov, V., I. Fernández-Val and A. Galichon (2010). Quantile and probability curves without crossing.
  *Econometrica* 78(3), 1093–1125.
- Genz, A. (2004). Numerical computation of rectangular bivariate and trivariate normal and t probabilities.
  *Statistics and Computing* 14(3), 251–260.
- He, X., X. Pan, K. M. Tan and W.-X. Zhou (2023). Smoothed quantile regression with large-scale inference.
  *Journal of Econometrics* 232(2), 367–388.
- Koenker, R. and G. Bassett (1978). Regression quantiles. *Econometrica* 46(1), 33–50.
- Koenker, R. and B. J. Park (1996). An interior point algorithm for nonlinear quantile regression. *Journal of
  Econometrics* 71, 265–283.
- Ma, S. and M. R. Kosorok (2005). Robust semiparametric M-estimation and the weighted bootstrap. *Journal of
  Multivariate Analysis* 96(1), 190–217.
- Pereda-Fernández, S. (2024). Fast algorithms for quantile regression with selection. arXiv:2402.16693.
- Portnoy, S. and R. Koenker (1997). The Gaussian hare and the Laplacian tortoise: computability of squared-error
  versus absolute-error estimators. *Statistical Science* 12(4), 279–300.
- Powell, J. L. (1991). Estimation of monotonic regression models under quantile restrictions. In W. A. Barnett,
  J. Powell and G. Tauchen (eds), *Nonparametric and Semiparametric Methods in Econometrics and Statistics*,
  Cambridge University Press, 357–384.
