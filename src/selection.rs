//! selection.rs -- selection-corrected quantile regression: Rust replication code for Arellano and Bonhomme
//! (2017), translated from their MATLAB code Code_QRS.m ("Code for selection-corrected quantile regression by
//! Manuel Arellano and Stephane Bonhomme"), by way of the Python port in erweinstein/quantile_regression
//! (selection_qr.py), and the weighted estimator and weighted bootstrap of its qr_extensions.py.
//!
//! The model: Y* = X'beta(U), observed only when D = 1{V <= p(Z)}, Z = (X, B) with B excluded from the
//! outcome equation, and the dependence of (U, V) given by a copula C(u, v; rho). Among participants the
//! tau-th conditional quantile of Y is then X'beta(tau) at the "rotated" level G(tau, p(Z); rho) =
//! C(tau, p(Z); rho) / p(Z). Code_QRS.m estimates p(Z) by probit, chooses rho on a grid so that the moment
//! E[p(Z) (1{Y <= X'beta(tau)} - G)] = 0 holds at tau = .2, .4, .6, .8 (summed, then squared), and then fits
//! the rotated quantile regressions at tau = .05, ..., .95.
//!
//! What is parallel: the copula grid (each rho is an independent set of fits), the final quantiles, and the
//! bootstrap draws, with rayon. Each fit is deterministic, so the results do not depend on the number of
//! threads (set RAYON_NUM_THREADS to control it).
//!
//! References
//! - Arellano M, Bonhomme S (2017). Quantile selection models with an application to understanding changes in
//!   wage inequality. Econometrica 85(1), 1-28.
//! - Ma S, Kosorok MR (2005). Robust semiparametric M-estimation and the weighted bootstrap. Journal of
//!   Multivariate Analysis 96(1), 190-217.
//! - Pereda-Fernandez S (2024). Fast algorithms for quantile regression with selection. arXiv:2402.16693.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use crate::linalg::{dot, lu_solve, Matrix, Qr};
use crate::normal;
use crate::rq::{rq_unchecked, LpOptions, Method, QrError, Tau};
use rand::SeedableRng;
use rand_distr::{Distribution, Exp1};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// numpy's round(x, 2).
fn round2(v: f64) -> f64 {
    (v * 100.0).round_ties_even() / 100.0
}

/// Code_QRS.m's grid for rho: 99 equidistant values -.98, -.96, ..., .98.
pub fn default_vecrho() -> Vec<f64> {
    (0..99).map(|i| round2(-0.98 + 0.02 * i as f64)).collect()
}

/// Code_QRS.m's quantiles for beta(tau): .05, .10, ..., .95.
pub fn default_taus() -> Vec<f64> {
    (0..19).map(|i| round2(0.05 + 0.05 * i as f64)).collect()
}

/// The grids of Code_QRS.m (the defaults) or of the paper.
#[derive(Clone, Debug)]
pub struct SelectionOptions {
    /// quantiles whose moment conditions identify rho (Code_QRS.m: .2, .4, .6, .8)
    pub vectau: Vec<f64>,
    /// grid for rho (Code_QRS.m: 99 values from -.98 to .98)
    pub vecrho: Vec<f64>,
    /// quantiles at which beta(tau) is reported (.05, ..., .95)
    pub taus: Vec<f64>,
}

impl Default for SelectionOptions {
    fn default() -> Self {
        SelectionOptions { vectau: vec![0.2, 0.4, 0.6, 0.8], vecrho: default_vecrho(), taus: default_taus() }
    }
}

/// The estimates, as returned by selection_qr() in the Python port.
#[derive(Clone, Debug)]
pub struct SelectionFit {
    /// estimated copula parameter
    pub rho: f64,
    /// (1 + dim X) x len(taus): row 0 the intercept, a column per tau
    pub beta: Matrix,
    pub taus: Vec<f64>,
    /// the criterion over the rho grid
    pub objective: Vec<f64>,
    pub vecrho: Vec<f64>,
    /// probit coefficients (constant first)
    pub gamma: Vec<f64>,
    /// propensity scores for all n observations
    pub pz: Vec<f64>,
}

/// Probit maximum likelihood by Fisher scoring, with a constant added and optional observation weights
/// (MATLAB: gamma = glmfit(Z, D, 'binomial', 'link', 'probit')). Returns gamma = (constant, slopes).
pub fn probit(d: &[f64], z: &Matrix, w: Option<&[f64]>) -> Result<Vec<f64>, QrError> {
    let wm = z.with_intercept();
    let (n, k) = (wm.nrow, wm.ncol);
    let mut g = vec![0.0; k];
    for _ in 0..100 {
        let mut score = vec![0.0; k];
        let mut h = vec![0.0; n];
        let mut lam = vec![0.0; n];
        for i in 0..n {
            let xb = dot(wm.row(i), &g);
            let p = normal::cdf(xb).clamp(1e-12, 1.0 - 1e-12);
            let phi = normal::pdf(xb);
            let wi = w.map_or(1.0, |w| w[i]);
            lam[i] = wi * phi * (d[i] - p) / (p * (1.0 - p));
            h[i] = wi * phi * phi / (p * (1.0 - p));
        }
        for i in 0..n {
            for (s, x) in score.iter_mut().zip(wm.row(i)) {
                *s += x * lam[i];
            }
        }
        let hm = wm.weighted_gram(&h);
        let step = lu_solve(&hm, &score).ok_or_else(|| QrError::Singular("probit information matrix".into()))?;
        let mut mx = 0.0f64;
        for (gj, s) in g.iter_mut().zip(&step) {
            *gj += s;
            mx = mx.max(s.abs());
        }
        if mx < 1e-10 {
            break;
        }
    }
    Ok(g)
}

/// The Gaussian copula C(u, v; rho) = Phi_2(Phi^{-1}(u), Phi^{-1}(v); rho)
/// (MATLAB: copulacdf('Gaussian', [u v], rho)).
#[inline]
pub fn gaussian_copula_cdf(u: f64, v: f64, rho: f64) -> f64 {
    normal::bvn_cdf(normal::ppf(u), normal::ppf(v), rho)
}

/// Observation-specific quantile levels G_i = C(tau, p_i; rho) / p_i, kept in [eps, 1 - eps] with eps = 1e-6
/// (MATLAB: G = copulacdf('Gaussian', [tau*ones(N1,1) pZ1], rho) ./ pZ1; the solver needs 0 < G < 1, and
/// the MATLAB code passes values of 0 or 1 on silently).
pub fn rotated_tau(tau: f64, p: &[f64], rho: f64) -> Vec<f64> {
    let eps = 1e-6;
    p.iter().map(|&pi| (gaussian_copula_cdf(tau, pi, rho) / pi).clamp(eps, 1.0 - eps)).collect()
}

/// Weighted regression quantile argmin_b sum_i w_i rho_tau(y_i - x_i'b), w_i >= 0, by rescaling the rows of X
/// and y by w (the check function is positively homogeneous); observations with w_i = 0 are dropped. Solved
/// with the Cholesky form of the Newton systems, as weighted_rq in qr_extensions.py.
pub fn weighted_rq<'a>(x: &Matrix, y: &[f64], tau: impl Into<Tau<'a>>, w: &[f64]) -> Result<Vec<f64>, QrError> {
    weighted_rq_impl(x, y, tau.into(), w, true)
}

/// weighted_rq, optionally without the input checks (for the rotated regressions, whose inputs are checked once).
fn weighted_rq_impl(x: &Matrix, y: &[f64], tau: Tau, w: &[f64], check: bool) -> Result<Vec<f64>, QrError> {
    let keep: Vec<usize> = (0..x.nrow).filter(|&i| w[i] > 0.0).collect();
    let mut xs = x.select_rows(&keep);
    for (r, &i) in keep.iter().enumerate() {
        for v in &mut xs.data[r * xs.ncol..(r + 1) * xs.ncol] {
            *v *= w[i];
        }
    }
    let ys: Vec<f64> = keep.iter().map(|&i| y[i] * w[i]).collect();
    let ts: Vec<f64> = keep.iter().map(|&i| tau.at(i)).collect();
    if check {
        crate::rq::rq_fit(&xs, &ys, Tau::PerObs(&ts), LpOptions::default(), Method::Cholesky).map(|f| f.coef)
    } else {
        Ok(rq_unchecked(&xs, &ys, Tau::PerObs(&ts), LpOptions::default(), Method::Cholesky).coef)
    }
}

/// Participants' data, ready for the rotated regressions.
struct Participants {
    x1c: Matrix,
    y1: Vec<f64>,
    pz1: Vec<f64>,
    w1: Option<Vec<f64>>,
}

fn setup(d: &[f64], y: &[f64], x: &Matrix, b: &Matrix, w: Option<&[f64]>) -> Result<(Vec<f64>, Vec<f64>, Participants), QrError> {
    let n = d.len();
    if y.len() != n || x.nrow != n || b.nrow != n || w.is_some_and(|w| w.len() != n) {
        return Err(QrError::Shape("D, Y, X, B (and w) should have the same number of rows".into()));
    }
    // Estimate propensity score.  MATLAB: Z=[X B]; gamma=glmfit(...); pZ=normcdf(gamma(1)+Z*gamma(2:end));
    let z = x.hstack(b);
    let gamma = probit(d, &z, w)?;
    let pz: Vec<f64> = (0..n).map(|i| normal::cdf(gamma[0] + dot(z.row(i), &gamma[1..]))).collect();
    // Select participants (D = 1; the Python port converts D to integers first)
    let sel: Vec<usize> = (0..n).filter(|&i| d[i].trunc() == 1.0).collect();
    let x1c = x.select_rows(&sel).with_intercept();
    if sel.len() < x1c.ncol || Qr::new(&x1c, None).rank() < x1c.ncol {
        return Err(QrError::RankDeficient);
    }
    let part = Participants {
        y1: sel.iter().map(|&i| y[i]).collect(),
        pz1: sel.iter().map(|&i| pz[i]).collect(),
        w1: w.map(|w| sel.iter().map(|&i| w[i]).collect()),
        x1c,
    };
    Ok((gamma, pz, part))
}

impl Participants {
    /// One rotated quantile regression at level tau and copula parameter rho.
    fn fit(&self, tau: f64, rho: f64, method: Method) -> Vec<f64> {
        let g = rotated_tau(tau, &self.pz1, rho);
        match &self.w1 {
            None => rq_unchecked(&self.x1c, &self.y1, Tau::PerObs(&g), LpOptions::default(), method).coef,
            Some(w1) => weighted_rq_impl(&self.x1c, &self.y1, Tau::PerObs(&g), w1, false).expect("weighted rotated regression"),
        }
    }
    /// The criterion at rho: (sum over vectau of the moment)^2 (MATLAB: obj_rho(j) = obj^2).
    fn objective(&self, rho: f64, vectau: &[f64], method: Method) -> f64 {
        let n1 = self.y1.len();
        let mut obj = 0.0;
        for &tau in vectau {
            let g = rotated_tau(tau, &self.pz1, rho);
            let beta = match &self.w1 {
                None => rq_unchecked(&self.x1c, &self.y1, Tau::PerObs(&g), LpOptions::default(), method).coef,
                Some(w1) => weighted_rq_impl(&self.x1c, &self.y1, Tau::PerObs(&g), w1, false).expect("weighted rotated regression"),
            };
            let mut m = 0.0;
            for i in 0..n1 {
                let ind = if self.y1[i] <= dot(self.x1c.row(i), &beta) { 1.0 } else { 0.0 };
                m += self.w1.as_ref().map_or(1.0, |w| w[i]) * self.pz1[i] * (ind - g[i]);
            }
            // unweighted: np.mean(varphi1 * (...)); weighted: sum(w1 * ...) / sum(w1)
            obj += m / self.w1.as_ref().map_or(n1 as f64, |w| w.iter().sum());
        }
        obj * obj
    }
}

fn estimate(d: &[f64], y: &[f64], x: &Matrix, b: &Matrix, w: Option<&[f64]>, opt: &SelectionOptions, method: Method) -> Result<SelectionFit, QrError> {
    let (gamma, pz, part) = setup(d, y, x, b, w)?;
    // Objective function over the rho grid (in parallel)
    let objective: Vec<f64> = opt.vecrho.par_iter().map(|&rho| part.objective(rho, &opt.vectau, method)).collect();
    // Minimize the objective function (the first minimum, as np.argmin)
    let mut jbest = 0;
    for (j, v) in objective.iter().enumerate() {
        if *v < objective[jbest] {
            jbest = j;
        }
    }
    let rho = opt.vecrho[jbest];
    // Estimate selection-corrected quantile parameters beta(tau)
    let cols: Vec<Vec<f64>> = opt.taus.par_iter().map(|&tau| part.fit(tau, rho, method)).collect();
    let p = part.x1c.ncol;
    let mut beta = Matrix::zeros(p, opt.taus.len());
    for (j, c) in cols.iter().enumerate() {
        for k in 0..p {
            beta.data[k * opt.taus.len() + j] = c[k];
        }
    }
    Ok(SelectionFit { rho, beta, taus: opt.taus.clone(), objective, vecrho: opt.vecrho.clone(), gamma, pz })
}

/// Selection-corrected quantile regression (Code_QRS.m; selection_qr() in the Python port).
///
/// - d: participation indicator (1 = participant)
/// - y: outcome (values for d = 0 are not used; they may be NaN)
/// - x: covariates, not including the constant (n x k)
/// - b: excluded covariates, not including the constant (n x m); Z = (X, B)
pub fn selection_qr(d: &[f64], y: &[f64], x: &Matrix, b: &Matrix, opt: &SelectionOptions) -> Result<SelectionFit, QrError> {
    estimate(d, y, x, b, None, opt, Method::Qr)
}

/// The estimator with observation weights in every step (the probit, the moment condition for rho and the
/// rotated quantile regressions); selection_qr_weighted() in qr_extensions.py. w = None gives unit weights.
pub fn selection_qr_weighted(d: &[f64], y: &[f64], x: &Matrix, b: &Matrix, w: Option<&[f64]>, opt: &SelectionOptions) -> Result<SelectionFit, QrError> {
    let ones;
    let w = match w {
        Some(w) => w,
        None => {
            ones = vec![1.0; d.len()];
            &ones
        }
    };
    estimate(d, y, x, b, Some(w), opt, Method::Cholesky)
}

/// Draws and standard errors of the weighted (multiplier) bootstrap.
#[derive(Clone, Debug)]
pub struct SelectionBootstrap {
    pub estimate: SelectionFit,
    pub rho_draws: Vec<f64>,
    /// one (1 + dim X) x len(taus) matrix per draw
    pub beta_draws: Vec<Matrix>,
    pub rho_se: f64,
    /// (1 + dim X) x len(taus)
    pub beta_se: Matrix,
}

fn sd(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

/// The weighted (multiplier) bootstrap of selection_qr_weighted() with Exp(1) weights (Ma and Kosorok 2005), as
/// selection_qr_bootstrap() in qr_extensions.py. Draw r uses its own Xoshiro256++ stream seeded from (seed, r),
/// so the draws do not depend on the number of threads; they differ from the Python draws, which use numpy's
/// generator.
pub fn selection_qr_bootstrap(d: &[f64], y: &[f64], x: &Matrix, b: &Matrix, reps: usize, seed: u64, opt: &SelectionOptions) -> Result<SelectionBootstrap, QrError> {
    let est = selection_qr_weighted(d, y, x, b, None, opt)?;
    let n = d.len();
    let draws: Vec<Result<SelectionFit, QrError>> = (0..reps)
        .into_par_iter()
        .map(|r| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let w: Vec<f64> = (0..n).map(|_| Exp1.sample(&mut rng)).collect();
            selection_qr_weighted(d, y, x, b, Some(&w), opt)
        })
        .collect();
    let draws: Vec<SelectionFit> = draws.into_iter().collect::<Result<_, _>>()?;
    let rho_draws: Vec<f64> = draws.iter().map(|f| f.rho).collect();
    let (p, j) = (est.beta.nrow, est.beta.ncol);
    let mut beta_se = Matrix::zeros(p, j);
    for k in 0..p * j {
        let v: Vec<f64> = draws.iter().map(|f| f.beta.data[k]).collect();
        beta_se.data[k] = sd(&v);
    }
    Ok(SelectionBootstrap {
        rho_se: sd(&rho_draws),
        rho_draws,
        beta_draws: draws.into_iter().map(|f| f.beta).collect(),
        beta_se,
        estimate: est,
    })
}
