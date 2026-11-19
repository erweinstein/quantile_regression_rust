//! extensions.rs -- additions beyond rq.m, rqic.m, ipqr.m and Code_QRS.m, translated from qr_extensions.py in
//! erweinstein/quantile_regression: standard errors for a regression quantile, and rearrangement. (The weighted
//! regression quantile, the weighted selection-corrected estimator and its weighted bootstrap are in
//! selection.rs.)
//!
//! 1. rq_se(): "ker" is Powell's (1991) kernel sandwich as computed by quantreg's summary.rq(se = "ker")
//!    (Gaussian kernel; bandwidth from Hall and Sheather (1988), scaled by a robust residual scale); "iid" is
//!    Koenker and Bassett's (1978) formula with the sparsity estimated by a difference quotient of residual
//!    quantiles (Siddiqui 1960), a simpler version of quantreg's se = "iid".
//! 2. rearrange(): Chernozhukov, Fernandez-Val and Galichon's (2010) rearrangement: sorting an estimated
//!    quantile curve x'beta(tau) over the tau grid gives a monotone estimate that is never farther from a
//!    monotone truth, in any L_p norm (their Proposition 1).
//!
//! References
//! - Chernozhukov V, Fernandez-Val I, Galichon A (2010). Quantile and probability curves without crossing.
//!   Econometrica 78(3), 1093-1125.
//! - Hall P, Sheather SJ (1988). On the distribution of a studentized quantile. JRSS B 50(3), 381-391.
//! - Koenker R (2005). Quantile Regression. Cambridge University Press, section 3.4.
//! - Koenker R, Bassett G (1978). Regression quantiles. Econometrica 46(1), 33-50.
//! - Powell JL (1991). Estimation of monotonic regression models under quantile restrictions. In Barnett,
//!   Powell and Tauchen (eds), Nonparametric and Semiparametric Methods in Econometrics and Statistics,
//!   Cambridge University Press, 357-384.
//! - Siddiqui MM (1960). Distribution of quantiles in samples from a bivariate population. Journal of Research
//!   of the National Bureau of Standards B 64(3), 145-150.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use crate::linalg::{inverse, matmul, Matrix};
use crate::normal;
use crate::rq::{rq, QrError};

/// numpy's quantile with its default (linear) interpolation, R's type 7.
pub fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    quantile_sorted(&s, q)
}

/// The same for a sorted slice.
pub fn quantile_sorted(s: &[f64], q: f64) -> f64 {
    let h = (s.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (h - lo as f64) * (s[hi] - s[lo])
}

/// Standard deviation with divisor n - ddof (numpy's std).
pub fn std(v: &[f64], ddof: usize) -> f64 {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - ddof as f64)).sqrt()
}

/// The Hall-Sheather (1988) bandwidth on the probability scale, as in quantreg's bandwidth.rq(hs = TRUE).
pub fn hall_sheather(n: usize, tau: f64, alpha: f64) -> f64 {
    let x0 = normal::ppf(tau);
    let f0 = normal::pdf(x0);
    (n as f64).powf(-1.0 / 3.0) * normal::ppf(1.0 - alpha / 2.0).powf(2.0 / 3.0) * ((1.5 * f0 * f0) / (2.0 * x0 * x0 + 1.0)).powf(1.0 / 3.0)
}

/// The kind of standard error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeMethod {
    /// Koenker and Bassett's tau(1 - tau) s^2 (X'X)^{-1}, s = 1/f(F^{-1}(tau)) by a difference quotient
    Iid,
    /// Powell's kernel sandwich tau(1 - tau) H^{-1} (X'X/n) H^{-1} / n
    Ker,
}

/// Standard errors of rq(X, y, tau) and their covariance matrix (beta = None: fitted here with rq()).
pub fn rq_se(x: &Matrix, y: &[f64], tau: f64, beta: Option<&[f64]>, method: SeMethod, alpha: f64) -> Result<(Vec<f64>, Matrix), QrError> {
    let (n, p) = (x.nrow, x.ncol);
    let fitted;
    let beta = match beta {
        Some(b) => b,
        None => {
            fitted = rq(x, y, tau)?;
            &fitted
        }
    };
    let xb = x.mul_vec(beta);
    let u: Vec<f64> = y.iter().zip(&xb).map(|(a, b)| a - b).collect();
    let h = hall_sheather(n, tau, alpha);
    let (lo, hi) = ((tau - h).max(1e-6), (tau + h).min(1.0 - 1e-6));
    let mut xtx = x.weighted_gram(&vec![1.0; n]);
    for v in &mut xtx.data {
        *v /= n as f64;
    }
    let singular = || QrError::Singular("rq_se".into());
    let mut sorted = u.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cov = match method {
        SeMethod::Iid => {
            let s = (quantile_sorted(&sorted, hi) - quantile_sorted(&sorted, lo)) / (hi - lo);
            let mut c = inverse(&xtx).ok_or_else(singular)?;
            for v in &mut c.data {
                *v *= tau * (1.0 - tau) * s * s / n as f64;
            }
            c
        }
        SeMethod::Ker => {
            let (q75, q25) = (quantile_sorted(&sorted, 0.75), quantile_sorted(&sorted, 0.25));
            let k = (normal::ppf(tau + h) - normal::ppf(tau - h)) * std(&u, 1).min((q75 - q25) / 1.34);
            let f: Vec<f64> = u.iter().map(|ui| normal::pdf(ui / k) / k).collect();
            let mut hm = x.weighted_gram(&f);
            for v in &mut hm.data {
                *v /= n as f64;
            }
            let hi_m = inverse(&hm).ok_or_else(singular)?;
            let mut c = matmul(&matmul(&hi_m, &xtx), &hi_m);
            for v in &mut c.data {
                *v *= tau * (1.0 - tau) / n as f64;
            }
            c
        }
    };
    let se = (0..p).map(|j| cov.get(j, j).sqrt()).collect();
    Ok((se, cov))
}

/// Monotone (rearranged) conditional quantiles. beta: p x J quantile coefficients on an increasing tau grid
/// (a column per tau, first row the intercept, as SelectionFit::beta); x: m x p covariate rows including the
/// leading 1. Returns the m x J curves x'beta(tau_j), each sorted over j.
pub fn rearrange(beta: &Matrix, x: &Matrix) -> Matrix {
    let mut q = matmul(x, beta);
    let j = q.ncol;
    for i in 0..q.nrow {
        q.data[i * j..(i + 1) * j].sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_is_numpy_linear() {
        let v = [3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(quantile(&v, 0.5), 3.0);
        assert!((quantile(&v, 0.3) - 1.4).abs() < 1e-15); // numpy: 1.4
    }
}
