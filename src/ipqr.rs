//! ipqr.rs -- nonlinear quantile regression by Koenker and Park's (1996) interior point algorithm: a Rust
//! translation of ipqr.m and ipqr_objfunc.m, by way of the Python port in erweinstein/quantile_regression
//! (ipqr.py).
//!
//!     Koenker, R. and B. J. Park (1996), "An Interior Point Algorithm for Nonlinear Quantile Regression,"
//!     Journal of Econometrics 71, 265-283.
//!
//! The MATLAB version passes the model as the name of a function and finds its Jacobian by prepending 'd' to
//! that name; the Python port passes two functions. Here the model is any type implementing [`Model`]: its
//! fitted values f(x, beta) and its n x p Jacobian. [`Linear`] is the linear model X b, and [`FnModel`]
//! wraps two closures. The line search on t in [-1, 1] (MATLAB's fmin, later fminbnd; scipy's
//! minimize_scalar(method = 'bounded') in the Python port) is a translation of scipy 1.16.3's
//! _minimize_scalar_bounded, Brent's (1973) method with the same tolerances, so the iterates follow the Python
//! port's.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use crate::linalg::{lu_solve, Matrix};
use crate::rq::QrError;

/// A nonlinear regression function: fitted values f(x, beta) (n values) and their Jacobian with respect to
/// beta (n x p). The covariates are held by the implementing type.
pub trait Model {
    fn value(&self, beta: &[f64]) -> Vec<f64>;
    fn jacobian(&self, beta: &[f64]) -> Matrix;
}

/// The linear model f(x, b) = X b, whose Jacobian is X.
pub struct Linear<'a>(pub &'a Matrix);

impl Model for Linear<'_> {
    fn value(&self, beta: &[f64]) -> Vec<f64> {
        self.0.mul_vec(beta)
    }
    fn jacobian(&self, _beta: &[f64]) -> Matrix {
        self.0.clone()
    }
}

/// A model given by two closures, beta -> f and beta -> Jacobian.
pub struct FnModel<F, J>
where
    F: Fn(&[f64]) -> Vec<f64>,
    J: Fn(&[f64]) -> Matrix,
{
    pub f: F,
    pub jac: J,
}

impl<F, J> Model for FnModel<F, J>
where
    F: Fn(&[f64]) -> Vec<f64>,
    J: Fn(&[f64]) -> Matrix,
{
    fn value(&self, beta: &[f64]) -> Vec<f64> {
        (self.f)(beta)
    }
    fn jacobian(&self, beta: &[f64]) -> Matrix {
        (self.jac)(beta)
    }
}

/// Options of ipqr: toler (minimum change in the objective before convergence is declared), eta (step-length
/// factor; 0.97 is Koenker and Park's value), inner (inner iterations per outer iteration; they recommend 2)
/// and max_iter (a safeguard on the outer iterations, not in the MATLAB code).
#[derive(Clone, Copy, Debug)]
pub struct IpqrOptions {
    pub toler: f64,
    pub eta: f64,
    pub inner: usize,
    pub max_iter: usize,
}

impl Default for IpqrOptions {
    fn default() -> Self {
        IpqrOptions { toler: 1e-8, eta: 0.97, inner: 2, max_iter: 500 }
    }
}

#[derive(Clone, Debug)]
pub struct IpqrFit {
    pub beta: Vec<f64>,
    pub objective: f64,
    pub iterations: usize,
}

/// sum of rho_q(res) = q sum(res) - sum(res[res < 0]) (ipqr_objfunc.m).
pub fn qr_objective(res: &[f64], q: f64) -> f64 {
    q * res.iter().sum::<f64>() - res.iter().filter(|&&r| r < 0.0).sum::<f64>()
}

/// Minimize a function of one variable on [x1, x2] by Brent's method, as scipy's
/// minimize_scalar(method = 'bounded') with its defaults (xatol = 1e-5, maxiter = 500). Returns the minimizer.
pub fn fminbound<F: FnMut(f64) -> f64>(mut func: F, x1: f64, x2: f64, xatol: f64, maxiter: usize) -> f64 {
    let sqrt_eps = (2.2e-16f64).sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let (mut rat, mut e) = (0.0f64, 0.0f64);
    let mut x;
    let mut fx = func(xf);
    let mut num = 1;
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;
    let sign = |v: f64| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 };
    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        // Check for parabolic fit
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            // Check for acceptability of parabola
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = (p + 0.0) / q;
                x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = sign(xm - xf) + if xm - xf == 0.0 { 1.0 } else { 0.0 };
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            // do a golden-section step
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = sign(rat) + if rat == 0.0 { 1.0 } else { 0.0 };
        x = xf + si * rat.abs().max(tol1);
        let fu = func(x);
        num += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= maxiter {
            break;
        }
    }
    xf
}

/// Nonlinear regression quantile by the interior point algorithm (ipqr.m), with Koenker and Park's defaults.
/// q: the quantile, in (0, 1); y: the response; model: fitted values and Jacobian; beta: starting value.
pub fn ipqr(q: f64, y: &[f64], model: &dyn Model, beta: &[f64]) -> Result<IpqrFit, QrError> {
    ipqr_with(q, y, model, beta, IpqrOptions::default())
}

/// ipqr with options.
pub fn ipqr_with(q: f64, y: &[f64], model: &dyn Model, beta: &[f64], opt: IpqrOptions) -> Result<IpqrFit, QrError> {
    if !(q > 0.0 && q < 1.0) {
        return Err(QrError::Tau("q should be strictly between 0 and 1".into()));
    }
    let n = y.len();
    let mut beta = beta.to_vec();
    let mut d = vec![0.0; n];
    let resid = |b: &[f64]| -> Vec<f64> { y.iter().zip(model.value(b)).map(|(yi, fi)| yi - fi).collect() };
    let mut residuals = resid(&beta);
    if residuals.len() != n {
        return Err(QrError::Shape("the model's fitted values should have the length of y".into()));
    }
    let mut lastobj = qr_objective(&residuals, q);
    let mut change = f64::INFINITY;
    let mut iteration = 0;
    let singular = || QrError::Singular("ipqr: J' W J".into());

    while change.abs() > opt.toler && iteration < opt.max_iter {
        iteration += 1;
        let j = model.jacobian(&beta); // J is the first-derivative matrix
        // MATLAB: dhat = d - J*(inv(J'*J)*(J'*d));   projection of d off the columns of J
        let jtj = j.weighted_gram(&vec![1.0; n]);
        let proj = lu_solve(&jtj, &j.t_mul_vec(&d)).ok_or_else(singular)?;
        let jp = j.mul_vec(&proj);
        let dhat: Vec<f64> = d.iter().zip(&jp).map(|(a, b)| a - b).collect();
        // MATLAB: m = toler + max([-dhat./(1-q); dhat./q]);
        let mut mx = f64::NEG_INFINITY;
        for &v in &dhat {
            mx = mx.max(-v / (1.0 - q)).max(v / q);
        }
        let m = opt.toler + mx;
        for (di, v) in d.iter_mut().zip(&dhat) {
            *di = v / m;
        }

        // Inner iterations (K&P recommend two per outer iteration).
        let mut direc = vec![0.0; beta.len()];
        for _ in 0..opt.inner {
            // MATLAB: dmatrix = min([q-d 1-q+d]')';  d2 = dmatrix.^2;
            let d2: Vec<f64> = d.iter().map(|&di| (q - di).min(1.0 - q + di).powi(2)).collect();
            // MATLAB: direc = inv(J'*(d2(:,ones(p,1)).*J))*(J'*(d2.*residuals));
            let g = j.weighted_gram(&d2);
            let rhs = j.t_mul_vec(&d2.iter().zip(&residuals).map(|(a, b)| a * b).collect::<Vec<_>>());
            direc = lu_solve(&g, &rhs).ok_or_else(singular)?;
            let jd = j.mul_vec(&direc);
            let s: Vec<f64> = (0..n).map(|i| d2[i] * (residuals[i] - jd[i])).collect();
            let mut alpha = f64::NEG_INFINITY;
            for i in 0..n {
                alpha = alpha.max(s[i] / (q - d[i])).max(-s[i] / (1.0 - q + d[i]));
            }
            for i in 0..n {
                d[i] += opt.eta / alpha * s[i];
            }
        }

        // Line search recommended by K&P, on t in [-1, 1] along direc.
        let t = fminbound(
            |t| {
                let b: Vec<f64> = beta.iter().zip(&direc).map(|(bi, di)| bi + t * di).collect();
                qr_objective(&resid(&b), q)
            },
            -1.0,
            1.0,
            1e-5,
            500,
        );
        for (bi, di) in beta.iter_mut().zip(&direc) {
            *bi += t * di;
        }
        residuals = resid(&beta);
        let obj = qr_objective(&residuals, q);
        change = lastobj - obj;
        lastobj = obj;
    }
    Ok(IpqrFit { beta, objective: lastobj, iterations: iteration })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fminbound_finds_a_parabola_minimum() {
        let t = fminbound(|t| (t - 0.3) * (t - 0.3) + 1.0, -1.0, 1.0, 1e-5, 500);
        assert!((t - 0.3).abs() < 1e-5);
    }
}
