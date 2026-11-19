//! smoothed.rs -- convolution-smoothed quantile regression ("conquer"; Fernandes, Guerre and Horta 2021; He,
//! Pan, Tan and Zhou 2023) with a multiplier bootstrap, translated from smoothed_qr.py in
//! erweinstein/quantile_regression.
//!
//! Smoothing the check function by convolution with a Gaussian kernel K_h gives the convex, twice
//! differentiable loss l_h(u) = E rho_tau(u + h Z) = h phi(u/h) + u (tau - Phi(-u/h)), with derivative
//! tau - Phi(-u/h). The estimator beta_h = argmin_b (1/n) sum_i l_h(y_i - x_i'b) is found by damped Newton steps
//! (p is small here; the conquer package uses gradient descent with Barzilai-Borwein steps). It is a different
//! estimator from rq() (it depends on h), so it is offered beside rq(), not instead of it. Default bandwidth as
//! in the conquer R package: h = max(((log n + p) / n)^0.4, 0.05).
//!
//! References
//! - Fernandes M, Guerre E, Horta E (2021). Smoothing quantile regressions. Journal of Business & Economic
//!   Statistics 39(1), 338-357.
//! - He X, Pan X, Tan KM, Zhou W-X (2023). Smoothed quantile regression with large-scale inference. Journal of
//!   Econometrics 232(2), 367-388.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use crate::extensions::quantile_sorted;
use crate::linalg::{dot, lstsq, lu_solve, Matrix};
use crate::normal;
use rand::SeedableRng;
use rand_distr::{Distribution, Exp1};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// The conquer package's default bandwidth, max(((log n + p) / n)^0.4, 0.05).
pub fn default_bandwidth(n: usize, p: usize) -> f64 {
    (((n as f64).ln() + p as f64) / n as f64).powf(0.4).max(0.05)
}

/// Smoothed quantile regression (Gaussian kernel) by damped Newton steps. X is n x p (include the intercept
/// column yourself, as for rq); h = None uses default_bandwidth; w are optional nonnegative weights (the
/// multiplier bootstrap uses them); b0 a starting value (None: weighted least squares). tol = 1e-8 and
/// max_iter = 100 in the Python port.
pub fn conquer(x: &Matrix, y: &[f64], tau: f64, h: Option<f64>, w: Option<&[f64]>, b0: Option<&[f64]>, tol: f64, max_iter: usize) -> Vec<f64> {
    let (n, p) = (x.nrow, x.ncol);
    let h = h.unwrap_or_else(|| default_bandwidth(n, p));
    let ones = vec![1.0; n];
    let w = w.unwrap_or(&ones);
    let mut b = match b0 {
        Some(b) => b.to_vec(),
        None => {
            let sw: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
            let ys: Vec<f64> = y.iter().zip(&sw).map(|(a, s)| a * s).collect();
            lstsq(x, Some(&sw), &ys)
        }
    };
    let wloss = |bb: &[f64]| -> f64 {
        let xb = x.mul_vec(bb);
        (0..n)
            .map(|i| {
                let r = y[i] - xb[i];
                w[i] * (h * normal::pdf(r / h) + r * (tau - normal::cdf(-r / h)))
            })
            .sum::<f64>()
            / n as f64
    };
    let mut f = wloss(&b);
    for _ in 0..max_iter {
        let xb = x.mul_vec(&b);
        let r: Vec<f64> = y.iter().zip(&xb).map(|(a, c)| a - c).collect();
        let gv: Vec<f64> = (0..n).map(|i| w[i] * (normal::cdf(-r[i] / h) - tau)).collect();
        let g: Vec<f64> = x.t_mul_vec(&gv).iter().map(|v| v / n as f64).collect();
        let hw: Vec<f64> = (0..n).map(|i| w[i] * normal::pdf(r[i] / h) / h).collect();
        let mut hm = x.weighted_gram(&hw);
        for v in &mut hm.data {
            *v /= n as f64;
        }
        for j in 0..p {
            hm.data[j * p + j] += 1e-12;
        }
        let step = lu_solve(&hm, &g).unwrap_or_else(|| g.clone());
        let mut t = 1.0;
        let (mut bn, mut fnew);
        loop {
            // backtracking line search (the loss is convex)
            bn = b.iter().zip(&step).map(|(bi, si)| bi - t * si).collect::<Vec<_>>();
            fnew = wloss(&bn);
            if fnew <= f - 1e-4 * t * dot(&g, &step) || t < 1e-10 {
                break;
            }
            t /= 2.0;
        }
        b = bn;
        f = fnew;
        if step.iter().map(|s| (t * s).abs()).fold(0.0, f64::max) < tol {
            break;
        }
    }
    b
}

/// The multiplier bootstrap of conquer, with its intervals.
#[derive(Clone, Debug)]
pub struct ConquerBootstrap {
    pub estimate: Vec<f64>,
    /// one row per draw
    pub draws: Vec<Vec<f64>>,
    pub se: Vec<f64>,
    pub h: f64,
    /// (lower, upper) per coefficient
    pub percentile: Vec<(f64, f64)>,
    pub pivotal: Vec<(f64, f64)>,
    pub normal: Vec<(f64, f64)>,
}

/// Multiplier bootstrap: refit with Exp(1) weights (mean 1, variance 1), each refit started from the
/// full-sample estimate, with percentile, pivotal and normal intervals (as conquer's ci = "bootstrap"). Draw r
/// uses its own Xoshiro256++ stream seeded from (seed, r); the draws run in parallel.
pub fn conquer_bootstrap(x: &Matrix, y: &[f64], tau: f64, h: Option<f64>, reps: usize, alpha: f64, seed: u64) -> ConquerBootstrap {
    let (n, p) = (x.nrow, x.ncol);
    let h = h.unwrap_or_else(|| default_bandwidth(n, p));
    let b = conquer(x, y, tau, Some(h), None, None, 1e-8, 100);
    let draws: Vec<Vec<f64>> = (0..reps)
        .into_par_iter()
        .map(|r| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let w: Vec<f64> = (0..n).map(|_| Exp1.sample(&mut rng)).collect();
            conquer(x, y, tau, Some(h), Some(&w), Some(&b), 1e-8, 100)
        })
        .collect();
    let z = normal::ppf(1.0 - alpha / 2.0);
    let (mut se, mut perc, mut piv, mut nor) = (vec![], vec![], vec![], vec![]);
    for j in 0..p {
        let mut v: Vec<f64> = draws.iter().map(|d| d[j]).collect();
        v.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let (lo, hi) = (quantile_sorted(&v, alpha / 2.0), quantile_sorted(&v, 1.0 - alpha / 2.0));
        let s = crate::extensions::std(&v, 1);
        se.push(s);
        perc.push((lo, hi));
        piv.push((2.0 * b[j] - hi, 2.0 * b[j] - lo));
        nor.push((b[j] - z * s, b[j] + z * s));
    }
    ConquerBootstrap { estimate: b, draws, se, h, percentile: perc, pivotal: piv, normal: nor }
}
