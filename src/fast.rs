//! fast.rs -- faster computation of Arellano and Bonhomme's estimator, translated from selection_qr_fast.py in
//! erweinstein/quantile_regression. (The Cholesky form of the Newton systems, its lp_fnm_fast, is
//! rq::Method::Cholesky.)
//!
//! 1. rq_preprocessed(): Portnoy and Koenker's (1997) preprocessing, written for observation-specific quantile
//!    levels tau_i. Given a preliminary estimate b0 (here the solution at the neighbouring copula value), only
//!    the M observations with the smallest residuals are kept; the others are summarized by two
//!    pseudo-observations ("globs"), one for those expected above the fitted line and one for those below.
//!    The reduced problem is solved and the signs of the globbed observations' residuals are checked; any that
//!    are wrong are added back and the problem is solved again. When every sign is right, the reduced linear
//!    program has the same solution as the full one.
//! 2. selection_qr_fast(): the estimator computed with (1) optionally, sweeping the copula grid in order so
//!    each fit starts from the previous one (the warm starts of Pereda-Fernandez 2024, Algorithm 1), and
//!    optionally evaluating the copula objective on a coarse grid first and then on the full grid only around
//!    the best coarse values (in the spirit of his Algorithm 3), which is exact when the overall minimizer lies
//!    in one of those neighbourhoods and is not guaranteed. Settings::Paper uses the paper's choices for the
//!    copula search (the nine deciles and 200 grid points; Arellano and Bonhomme 2017, section 5) instead of
//!    Code_QRS.m's.
//!
//! One caveat (as in the Python port): p observations lie exactly on each fitted line, and in floating point
//! they land slightly above or below it depending on the path the solver took. Code_QRS.m's moment
//! 1{Y <= X'b} counts them by that rounding, so preprocessing can move the estimated copula parameter by a
//! grid step or two even though every regression reaches the same optimum. It is therefore an option, off by
//! default; tie_tol > 0 counts residuals within tie_tol (1 + sd of Y among participants) of zero as Y <= X'b,
//! which makes the result independent of that rounding.
//!
//! When preprocessing is off, the copula grid is evaluated in parallel (the fits are independent); with it on,
//! the sweep is sequential, because each fit starts from the previous one.
//!
//! References
//! - Arellano M, Bonhomme S (2017). Quantile selection models with an application to understanding changes in
//!   wage inequality. Econometrica 85(1), 1-28.
//! - Pereda-Fernandez S (2024). Fast algorithms for quantile regression with selection. arXiv:2402.16693.
//! - Portnoy S, Koenker R (1997). The Gaussian hare and the Laplacian tortoise: computability of squared-error
//!   versus absolute-error estimators. Statistical Science 12(4), 279-300.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use crate::extensions::std;
use crate::linalg::{dot, Matrix, Qr};
use crate::normal;
use crate::rq::{rq_unchecked, LpOptions, Method, QrError, Tau};
use crate::selection::{default_taus, default_vecrho, probit, rotated_tau_q};
use rayon::prelude::*;

fn rq_fast_unchecked(x: &Matrix, y: &[f64], tau: Tau) -> Vec<f64> {
    rq_unchecked(x, y, tau, LpOptions::default(), Method::Cholesky).coef
}

/// What rq_preprocessed did: the rounds, the size of the final reduced problem, and whether it fell back to
/// the full problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreprocessInfo {
    pub rounds: usize,
    pub kept: usize,
    pub fallback: bool,
}

/// Regression quantile of y on X at levels tau (one number or one per observation), using b0 to set aside the
/// observations whose residual signs are predictable (Portnoy and Koenker 1997). m = None: the Python port's
/// default, min(n, max(10 p, ceil(3 sqrt(n p)))); max_rounds = 8 and theta = 0.5 in rq_preprocessed_default.
pub fn rq_preprocessed(x: &Matrix, y: &[f64], tau: Tau, b0: &[f64], m: Option<usize>, max_rounds: usize, theta: f64) -> (Vec<f64>, PreprocessInfo) {
    let (n, p) = (x.nrow, x.ncol);
    let m = m.unwrap_or_else(|| n.min((10 * p).max((3.0 * ((n * p) as f64).sqrt()).ceil() as usize)));
    if m >= n {
        return (rq_fast_unchecked(x, y, tau), PreprocessInfo { rounds: 0, kept: n, fallback: true });
    }
    let xb0 = x.mul_vec(b0);
    let r0: Vec<f64> = y.iter().zip(&xb0).map(|(a, b)| a - b).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| r0[i].abs().partial_cmp(&r0[j].abs()).unwrap());
    let mut keep = vec![false; n];
    for &i in &order[..m] {
        keep[i] = true;
    }
    let scale = std(y, 0) + 1.0;
    let colsum_max = (0..p).map(|j| (0..n).map(|i| x.get(i, j).abs()).sum::<f64>()).fold(f64::NEG_INFINITY, f64::max);
    let big = 1e3 * scale * (1.0 + colsum_max);
    for rnd in 1..=max_rounds {
        let up: Vec<bool> = (0..n).map(|i| !keep[i] && r0[i] > 0.0).collect();
        let lo: Vec<bool> = (0..n).map(|i| !keep[i] && r0[i] <= 0.0).collect();
        let idx: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
        let mut xr = x.select_rows(&idx);
        let mut yr: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let mut tr: Vec<f64> = idx.iter().map(|&i| tau.at(i)).collect();
        let mut glob = |sel: &[bool], wt: &dyn Fn(usize) -> f64, denom: f64, sign: f64| {
            if sel.iter().any(|&s| s) {
                let mut xg = vec![0.0; p];
                for i in (0..n).filter(|&i| sel[i]) {
                    let w = wt(i);
                    for (g, v) in xg.iter_mut().zip(x.row(i)) {
                        *g += w * v;
                    }
                }
                for g in &mut xg {
                    *g /= denom;
                }
                yr.push(dot(&xg, b0) + sign * big);
                xr.data.extend_from_slice(&xg);
                xr.nrow += 1;
                tr.push(theta);
            }
        };
        glob(&up, &|i| tau.at(i), theta, 1.0);
        glob(&lo, &|i| 1.0 - tau.at(i), 1.0 - theta, -1.0);
        let b = rq_fast_unchecked(&xr, &yr, Tau::PerObs(&tr));
        let xb = x.mul_vec(&b);
        let mut any_bad = false;
        let mut bad = vec![false; n];
        for i in 0..n {
            let r = y[i] - xb[i];
            bad[i] = (up[i] && r < 0.0) || (lo[i] && r > 0.0);
            any_bad |= bad[i];
        }
        if !any_bad {
            return (b, PreprocessInfo { rounds: rnd, kept: keep.iter().filter(|&&k| k).count(), fallback: false });
        }
        // add the wrongly signed observations and, as Portnoy and Koenker do, widen the band
        for i in 0..n {
            keep[i] |= bad[i];
        }
        let extra: Vec<usize> = order.iter().copied().filter(|&i| !keep[i]).take(m).collect();
        for i in extra {
            keep[i] = true;
        }
        if keep.iter().filter(|&&k| k).count() >= n {
            break;
        }
    }
    (rq_fast_unchecked(x, y, tau), PreprocessInfo { rounds: max_rounds, kept: n, fallback: true })
}

/// rq_preprocessed with the Python port's defaults (m from n and p, max_rounds = 8, theta = 0.5).
pub fn rq_preprocessed_default(x: &Matrix, y: &[f64], tau: Tau, b0: &[f64]) -> (Vec<f64>, PreprocessInfo) {
    rq_preprocessed(x, y, tau, b0, None, 8, 0.5)
}

/// Code_QRS.m's copula search (four quantiles .2-.8, 99 grid points) or the paper's (the nine deciles, 200
/// points spaced evenly over (-0.995, 0.995); the paper does not give the grid's end points).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Settings {
    Code,
    Paper,
}

/// numpy's linspace(start, stop, num).
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    v[num - 1] = stop;
    v
}

impl Settings {
    pub fn vectau(self) -> Vec<f64> {
        match self {
            Settings::Code => vec![0.2, 0.4, 0.6, 0.8],
            Settings::Paper => (1..=9).map(|i| ((i as f64 * 0.1) * 10.0).round() / 10.0).collect(),
        }
    }
    pub fn vecrho(self) -> Vec<f64> {
        match self {
            Settings::Code => default_vecrho(),
            Settings::Paper => linspace(-0.995, 0.995, 200),
        }
    }
}

/// How the copula grid is searched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Search {
    /// every grid point (the same estimate as selection_qr with the same grid)
    Full,
    /// every coarse_step-th point first, then the full grid around the n_best best coarse points
    Coarse { coarse_step: usize, n_best: usize },
}

/// Options of selection_qr_fast; Default is the Python port's (Code_QRS.m's grids, no preprocessing, full
/// search, tie_tol = 0).
#[derive(Clone, Debug)]
pub struct FastOptions {
    pub settings: Settings,
    /// overrides the settings' vectau if given
    pub vectau: Option<Vec<f64>>,
    /// overrides the settings' vecrho if given
    pub vecrho: Option<Vec<f64>>,
    pub taus: Vec<f64>,
    pub preprocess: bool,
    pub search: Search,
    pub tie_tol: f64,
}

impl Default for FastOptions {
    fn default() -> Self {
        FastOptions { settings: Settings::Code, vectau: None, vecrho: None, taus: default_taus(), preprocess: false, search: Search::Full, tie_tol: 0.0 }
    }
}

impl Search {
    /// The Python port's coarse search: coarse_step = 5, n_best = 3.
    pub fn coarse() -> Search {
        Search::Coarse { coarse_step: 5, n_best: 3 }
    }
}

/// The estimates of selection_qr_fast: as SelectionFit, with NaN in `objective` where the grid was not
/// evaluated, and the number of grid points evaluated.
#[derive(Clone, Debug)]
pub struct FastFit {
    pub rho: f64,
    pub beta: Matrix,
    pub taus: Vec<f64>,
    pub objective: Vec<f64>,
    pub vecrho: Vec<f64>,
    pub gamma: Vec<f64>,
    pub pz: Vec<f64>,
    pub evaluated: usize,
}

/// Arellano and Bonhomme's estimator, as selection::selection_qr, computed faster (selection_qr_fast in the
/// Python port). Inputs as selection_qr.
pub fn selection_qr_fast(d: &[f64], y: &[f64], x: &Matrix, b: &Matrix, opt: &FastOptions) -> Result<FastFit, QrError> {
    let vectau = opt.vectau.clone().unwrap_or_else(|| opt.settings.vectau());
    let vecrho = opt.vecrho.clone().unwrap_or_else(|| opt.settings.vecrho());
    let n = d.len();
    if y.len() != n || x.nrow != n || b.nrow != n {
        return Err(QrError::Shape("D, Y, X, B should have the same number of rows".into()));
    }
    let z = x.hstack(b);
    let gamma = probit(d, &z, None)?;
    let pz: Vec<f64> = (0..n).map(|i| normal::cdf(gamma[0] + dot(z.row(i), &gamma[1..]))).collect();
    let sel: Vec<usize> = (0..n).filter(|&i| d[i].trunc() == 1.0).collect();
    let y1: Vec<f64> = sel.iter().map(|&i| y[i]).collect();
    let pz1: Vec<f64> = sel.iter().map(|&i| pz[i]).collect();
    let q1: Vec<f64> = pz1.iter().map(|&p| normal::ppf(p)).collect();
    let x1c = x.select_rows(&sel).with_intercept();
    if sel.len() < x1c.ncol || Qr::new(&x1c, None).rank() < x1c.ncol {
        return Err(QrError::RankDeficient);
    }
    let tie = opt.tie_tol * (1.0 + std(&y1, 0));
    let n1 = y1.len() as f64;

    // the objective at one rho, with optional warm starts; returns (objective, solutions)
    let objective_at = |rhoa: f64, starts: &[Option<Vec<f64>>]| -> (f64, Vec<Option<Vec<f64>>>) {
        let mut obj = 0.0;
        let mut sols = Vec::with_capacity(vectau.len());
        for (k, &tau) in vectau.iter().enumerate() {
            let g = rotated_tau_q(tau, &pz1, &q1, rhoa);
            let beta = match (&starts[k], opt.preprocess) {
                (Some(s), true) => rq_preprocessed_default(&x1c, &y1, Tau::PerObs(&g), s).0,
                _ => rq_fast_unchecked(&x1c, &y1, Tau::PerObs(&g)),
            };
            let mut m = 0.0;
            for i in 0..y1.len() {
                let ind = if y1[i] <= dot(x1c.row(i), &beta) + tie { 1.0 } else { 0.0 };
                m += pz1[i] * (ind - g[i]);
            }
            obj += m / n1;
            sols.push(Some(beta));
        }
        (obj * obj, sols)
    };

    let jn = vecrho.len();
    let mut obj_rho = vec![f64::NAN; jn];
    let step = match opt.search {
        Search::Full => 1,
        Search::Coarse { coarse_step, .. } => coarse_step,
    };
    let sweep = |idx: &[usize], obj_rho: &mut Vec<f64>| {
        let mut todo: Vec<usize> = idx.iter().copied().filter(|&j| obj_rho[j].is_nan()).collect();
        todo.sort_unstable();
        if !opt.preprocess {
            // independent fits: evaluate in parallel (identical to the sequential sweep without warm starts)
            let vals: Vec<f64> = todo.par_iter().map(|&j| objective_at(vecrho[j], &vec![None; vectau.len()]).0).collect();
            for (j, v) in todo.iter().zip(vals) {
                obj_rho[*j] = v;
            }
            return;
        }
        let mut starts: Vec<Option<Vec<f64>>> = vec![None; vectau.len()];
        let mut prev: Option<usize> = None;
        for j in todo {
            if prev.is_none_or(|p| j - p > step) {
                starts = vec![None; vectau.len()]; // not adjacent: no warm start
            }
            let (v, s) = objective_at(vecrho[j], &starts);
            obj_rho[j] = v;
            starts = s;
            prev = Some(j);
        }
    };

    match opt.search {
        Search::Full => sweep(&(0..jn).collect::<Vec<_>>(), &mut obj_rho),
        Search::Coarse { coarse_step, n_best } => {
            let mut coarse: Vec<usize> = (0..jn).step_by(coarse_step).collect();
            if (jn - 1) % coarse_step != 0 {
                coarse.push(jn - 1);
            }
            sweep(&coarse, &mut obj_rho);
            let mut ranked = coarse.clone();
            ranked.sort_by(|&a, &b| obj_rho[a].partial_cmp(&obj_rho[b]).unwrap());
            let mut fine: Vec<usize> = ranked
                .iter()
                .take(n_best)
                .flat_map(|&j| (j.saturating_sub(coarse_step - 1))..(jn.min(j + coarse_step)))
                .collect();
            fine.sort_unstable();
            fine.dedup();
            sweep(&fine, &mut obj_rho);
        }
    }
    let evaluated = obj_rho.iter().filter(|v| !v.is_nan()).count();
    // np.nanargmin: the first minimum among the evaluated points
    let mut jbest = None;
    for (j, v) in obj_rho.iter().enumerate() {
        if !v.is_nan() && jbest.is_none_or(|b: usize| *v < obj_rho[b]) {
            jbest = Some(j);
        }
    }
    let rho = vecrho[jbest.expect("at least one grid point evaluated")];

    let p = x1c.ncol;
    let jt = opt.taus.len();
    let cols: Vec<Vec<f64>> = if opt.preprocess {
        let mut prev: Option<Vec<f64>> = None;
        let mut cols = Vec::with_capacity(jt);
        for &tau in &opt.taus {
            let g = rotated_tau_q(tau, &pz1, &q1, rho);
            let bcol = match &prev {
                Some(s) => rq_preprocessed_default(&x1c, &y1, Tau::PerObs(&g), s).0,
                None => rq_fast_unchecked(&x1c, &y1, Tau::PerObs(&g)),
            };
            prev = Some(bcol.clone());
            cols.push(bcol);
        }
        cols
    } else {
        opt.taus.par_iter().map(|&tau| rq_fast_unchecked(&x1c, &y1, Tau::PerObs(&rotated_tau_q(tau, &pz1, &q1, rho)))).collect()
    };
    let mut beta = Matrix::zeros(p, jt);
    for (j, c) in cols.iter().enumerate() {
        for k in 0..p {
            beta.data[k * jt + j] = c[k];
        }
    }
    Ok(FastFit { rho, beta, taus: opt.taus.clone(), objective: obj_rho, vecrho, gamma, pz, evaluated })
}
