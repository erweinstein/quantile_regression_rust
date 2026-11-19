//! rq.rs -- quantile regression by the Frisch-Newton interior point method: a Rust translation of rq.m
//! (`rq_fnm` and its helpers `lp_fnm` and `bound`) and of rqic.m (the same algorithm with linear inequality
//! constraints R b >= r).
//!
//! The algorithm and the MATLAB code are those of Daniel Morillo and Roger Koenker (`lp_fnm`), translated from
//! Ox to MATLAB by Paul Eilers (1999), modified by Roger Koenker (2000-) and Paul Eilers (2004); rqic.m is by
//! Roger Koenker (2002). See Portnoy, S. and R. Koenker (1997), "The Gaussian hare and the Laplacian
//! tortoise," Statistical Science 12(4), 279-300, and http://www.econ.uiuc.edu/~roger/research/rq/rq.html.
//! This file follows the Python port in erweinstein/quantile_regression (rq.py) statement by statement; the
//! comments give the MATLAB line where the correspondence is not obvious.
//!
//! The tau-th regression quantile minimizes sum_i rho_tau(y_i - x_i'b), rho_tau(u) = u (tau - 1{u < 0}). Its
//! dual is the bounded-variable linear program max_a { y'a : X'a = (1 - tau) X'1, 0 <= a <= 1 }, and the
//! coefficients are minus the Lagrange multipliers of the equality constraints. `rq` sets up this dual and
//! `lp_fnm` solves it.
//!
//! Storage: the MATLAB code works with A = X' (p x n). Here the design X (n x p, row-major) is passed
//! instead, so "A x" is X'x and "y A" (MATLAB's row vector times A) is X y.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use crate::linalg::{cholesky_solve, dot, lu_solve, Matrix, Qr};
use std::fmt;

/// Errors for invalid inputs (the Python port raises ValueError for the same cases).
#[derive(Debug, Clone, PartialEq)]
pub enum QrError {
    Shape(String),
    Tau(String),
    RankDeficient,
    Singular(String),
}

impl fmt::Display for QrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QrError::Shape(s) => write!(f, "shape: {s}"),
            QrError::Tau(s) => write!(f, "tau: {s}"),
            QrError::RankDeficient => write!(f, "X does not have full column rank"),
            QrError::Singular(s) => write!(f, "singular system: {s}"),
        }
    }
}
impl std::error::Error for QrError {}

/// How the p x p Newton systems are solved. `Qr` is the translation of MATLAB's `dy = (AQ' \ rhs)'` (least
/// squares of an n x p system by QR, as rq.py does with numpy's lstsq); `Cholesky` forms the normal equations
/// and solves them by Cholesky (rq_fast in selection_qr_fast.py), falling back to QR if they are not
/// numerically positive definite. Both give the same iterates up to rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Qr,
    Cholesky,
}

/// Constants of `lp_fnm`: step-length damping, tolerance on the duality gap, maximum iterations. As the
/// original authors note, you can change them if you want to.
#[derive(Clone, Copy, Debug)]
pub struct LpOptions {
    pub beta: f64,
    pub small: f64,
    pub max_it: usize,
}

impl Default for LpOptions {
    /// rq.m's values: beta = 0.9995, small = 1e-5, max_it = 50.
    fn default() -> Self {
        LpOptions { beta: 0.9995, small: 1e-5, max_it: 50 }
    }
}

impl LpOptions {
    /// rqic.m's values: beta = 0.9995, small = 1e-8, max_it = 100.
    pub fn rqic_default() -> Self {
        LpOptions { beta: 0.9995, small: 1e-8, max_it: 100 }
    }
}

/// The dual solution of `lp_fnm` and how the iterations ended. (The Python port warns on non-convergence;
/// here it is reported.)
#[derive(Clone, Debug)]
pub struct LpSolution {
    pub y: Vec<f64>,
    pub iterations: usize,
    pub gap: f64,
    pub converged: bool,
}

/// Allowed step lengths: -x/dx where dx < 0, 1e20 (effectively infinity) elsewhere.
/// MATLAB: `b = 1e20 + 0 * x; f = find(dx < 0); b(f) = -x(f) ./ dx(f);`
pub fn bound(x: &[f64], dx: &[f64]) -> Vec<f64> {
    x.iter().zip(dx).map(|(&xi, &di)| if di < 0.0 { -xi / di } else { 1e20 }).collect()
}

/// min over the elements of bound(x, dx), in one pass and without allocating.
#[inline]
fn min_bound(x: &[f64], dx: &[f64]) -> f64 {
    let mut m = 1e20f64;
    for (&xi, &di) in x.iter().zip(dx) {
        if di < 0.0 {
            let b = -xi / di;
            if b < m {
                m = b;
            }
        }
    }
    m
}

/// min(beta * min over all pairs of bound(v, dv), 1): MATLAB's `fp = min(fx, fs); fp = min(min(beta * fp), 1);`
#[inline]
fn step_length(beta: f64, pairs: &[(&[f64], &[f64])]) -> f64 {
    let m = pairs.iter().map(|(v, dv)| min_bound(v, dv)).fold(f64::INFINITY, f64::min);
    (beta * m).min(1.0)
}

/// A factorization for the Newton systems of one iteration: given the scaling sq = sqrt(q), solve the least
/// squares problem (diag(sq) X) dy = rhs.
enum Newton {
    Qr(Qr),
    Normal(Matrix, Option<Qr>),
}

impl Newton {
    fn new(x: &Matrix, sq: &[f64], q: &[f64], method: Method) -> Newton {
        match method {
            Method::Qr => Newton::Qr(Qr::new(x, Some(sq))),
            Method::Cholesky => Newton::Normal(x.weighted_gram(q), None),
        }
    }
    /// dy for the right-hand side rhs (length n).
    fn solve(&mut self, x: &Matrix, sq: &[f64], rhs: &[f64]) -> Vec<f64> {
        match self {
            Newton::Qr(f) => f.solve(rhs),
            Newton::Normal(m, fallback) => {
                // (AQ AQ') dy = AQ rhs, AQ = X' diag(sq)
                let mut v = vec![0.0; x.ncol];
                for i in 0..x.nrow {
                    let a = sq[i] * rhs[i];
                    for (o, xv) in v.iter_mut().zip(x.row(i)) {
                        *o += xv * a;
                    }
                }
                match cholesky_solve(m, &v) {
                    Some(dy) => dy,
                    None => fallback.get_or_insert_with(|| Qr::new(x, Some(sq))).solve(rhs),
                }
            }
        }
    }
}

/// Solve the bounded-variable linear program min c'x s.t. A x = b, 0 < x < u by the Frisch-Newton method with
/// Mehrotra's predictor-corrector steps (lp_fnm in rq.m), with A = X' given as the n x p matrix X. An initial
/// feasible point x0 (A x0 = b, 0 < x0 < u) must be provided. Returns the dual solution y (p values).
pub fn lp_fnm(xm: &Matrix, c: &[f64], b: &[f64], u: &[f64], x0: &[f64], opt: LpOptions, method: Method) -> LpSolution {
    let (n, p) = (xm.nrow, xm.ncol);
    assert!(c.len() == n && u.len() == n && x0.len() == n && b.len() == p, "lp_fnm: inconsistent lengths");
    let mut x = x0.to_vec();

    // Generate initial feasible point
    let mut s: Vec<f64> = u.iter().zip(&x).map(|(ui, xi)| ui - xi).collect();
    // MATLAB: y = (A' \ c')';   least squares, A' = X is n x p
    let mut y = Qr::new(xm, None).solve(c);
    // MATLAB: r = c - y * A;  r = r + 0.001 * (r == 0);  z = r .* (r > 0);  w = z - r;
    let ya = xm.mul_vec(&y);
    let mut z = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let mut r = c[i] - ya[i];
        if r == 0.0 {
            r += 0.001;
        }
        z[i] = if r > 0.0 { r } else { 0.0 };
        w[i] = z[i] - r;
    }
    let gap_of = |x: &[f64], y: &[f64], w: &[f64]| dot(c, x) - dot(y, b) + dot(w, u);
    let mut gap = gap_of(&x, &y, &w);

    let (mut q, mut sq, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut dx, mut ds, mut dz, mut dw) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut rhs = vec![0.0; n];
    let mut it = 0;
    while gap > opt.small && it < opt.max_it {
        it += 1;

        // Compute affine step
        for i in 0..n {
            q[i] = 1.0 / (z[i] / x[i] + w[i] / s[i]);
            r[i] = z[i] - w[i];
            sq[i] = q[i].sqrt();
            rhs[i] = sq[i] * r[i];
        }
        let mut newton = Newton::new(xm, &sq, &q, method);
        let mut dy = newton.solve(xm, &sq, &rhs); // MATLAB: dy = (AQ' \ rhs)';
        let ady = xm.mul_vec(&dy);
        for i in 0..n {
            dx[i] = q[i] * (ady[i] - r[i]); // MATLAB: dx = q .* (dy * A - r)';
            ds[i] = -dx[i];
            dz[i] = -z[i] * (1.0 + dx[i] / x[i]);
            dw[i] = -w[i] * (1.0 + ds[i] / s[i]);
        }

        // Compute maximum allowable step lengths
        let mut fp = step_length(opt.beta, &[(&x, &dx), (&s, &ds)]);
        let mut fd = step_length(opt.beta, &[(&w, &dw), (&z, &dz)]);

        // If full step is feasible, take it. Otherwise modify it
        if fp.min(fd) < 1.0 {
            // Update mu
            let mut mu = dot(&z, &x) + dot(&w, &s);
            let mut g = 0.0;
            for i in 0..n {
                g += (z[i] + fd * dz[i]) * (x[i] + fp * dx[i]) + (w[i] + fd * dw[i]) * (s[i] + fp * ds[i]);
            }
            // MATLAB: mu = mu * (g / mu) ^3 / ( 2* n);
            mu = mu * (g / mu).powi(3) / (2.0 * n as f64);

            // Compute modified step
            let mut xi = vec![0.0; n];
            let mut dxdz = vec![0.0; n];
            let mut dsdw = vec![0.0; n];
            for i in 0..n {
                dxdz[i] = dx[i] * dz[i];
                dsdw[i] = ds[i] * dw[i];
                xi[i] = mu * (1.0 / x[i] - 1.0 / s[i]);
                rhs[i] += sq[i] * (dxdz[i] - dsdw[i] - xi[i]);
            }
            dy = newton.solve(xm, &sq, &rhs);
            let ady = xm.mul_vec(&dy);
            for i in 0..n {
                let xinv = 1.0 / x[i];
                let sinv = 1.0 / s[i];
                dx[i] = q[i] * (ady[i] + xi[i] - r[i] - dxdz[i] + dsdw[i]);
                ds[i] = -dx[i];
                dz[i] = mu * xinv - z[i] - xinv * z[i] * dx[i] - dxdz[i];
                dw[i] = mu * sinv - w[i] - sinv * w[i] * ds[i] - dsdw[i];
            }

            // Compute maximum allowable step lengths
            fp = step_length(opt.beta, &[(&x, &dx), (&s, &ds)]);
            fd = step_length(opt.beta, &[(&w, &dw), (&z, &dz)]);
        }

        // Take the step (on every iteration, whether or not the step was modified above)
        for i in 0..n {
            x[i] += fp * dx[i];
            s[i] += fp * ds[i];
            w[i] += fd * dw[i];
            z[i] += fd * dz[i];
        }
        for (yj, dyj) in y.iter_mut().zip(&dy) {
            *yj += fd * dyj;
        }
        gap = gap_of(&x, &y, &w);
    }
    LpSolution { converged: gap <= opt.small, y, iterations: it, gap }
}

/// A quantile level: one number for all observations, or one per observation (the "rotated" quantile
/// regression of Arellano and Bonhomme uses the latter).
#[derive(Clone, Copy, Debug)]
pub enum Tau<'a> {
    All(f64),
    PerObs(&'a [f64]),
}

impl<'a> From<f64> for Tau<'a> {
    fn from(t: f64) -> Self {
        Tau::All(t)
    }
}
impl<'a> From<&'a [f64]> for Tau<'a> {
    fn from(t: &'a [f64]) -> Self {
        Tau::PerObs(t)
    }
}
impl<'a> From<&'a Vec<f64>> for Tau<'a> {
    fn from(t: &'a Vec<f64>) -> Self {
        Tau::PerObs(t.as_slice())
    }
}

impl<'a> Tau<'a> {
    #[inline]
    pub fn at(&self, i: usize) -> f64 {
        match self {
            Tau::All(t) => *t,
            Tau::PerObs(v) => v[i],
        }
    }
    pub fn check(&self, n: usize) -> Result<(), QrError> {
        match self {
            Tau::All(t) if !(*t > 0.0 && *t < 1.0) => Err(QrError::Tau("tau should be strictly between 0 and 1".into())),
            Tau::PerObs(v) if v.len() != n => Err(QrError::Shape(format!("tau should have length n = {n}"))),
            Tau::PerObs(v) if v.iter().any(|t| !(*t > 0.0 && *t < 1.0)) => {
                Err(QrError::Tau("tau should be strictly between 0 and 1".into()))
            }
            _ => Ok(()),
        }
    }
}

/// A fitted regression quantile with the solver's report.
#[derive(Clone, Debug)]
pub struct RqFit {
    pub coef: Vec<f64>,
    pub iterations: usize,
    pub gap: f64,
    pub converged: bool,
}

/// Regression quantile by the Frisch-Newton interior point method (rq_fnm in rq.m), with full options.
/// X is n x p and should include a column of ones for an intercept (rq does not add one, as in the MATLAB
/// original). Checks the inputs as rq.py does (lengths, 0 < tau < 1, full column rank).
pub fn rq_fit<'a>(x: &Matrix, y: &[f64], tau: impl Into<Tau<'a>>, opt: LpOptions, method: Method) -> Result<RqFit, QrError> {
    let tau = tau.into();
    let (n, p) = (x.nrow, x.ncol);
    if y.len() != n {
        return Err(QrError::Shape(format!("y should have length n = {n}")));
    }
    tau.check(n)?;
    if n < p || Qr::new(x, None).rank() < p {
        return Err(QrError::RankDeficient);
    }
    Ok(rq_unchecked(x, y, tau, opt, method))
}

/// rq without the input checks (for inner loops whose inputs are known to be valid).
pub fn rq_unchecked(x: &Matrix, y: &[f64], tau: Tau, opt: LpOptions, method: Method) -> RqFit {
    let n = x.nrow;
    // MATLAB: u = ones(m, 1); a = (1 - p) .* u; b = -lp_fnm(X', -y', X' * a, u, a)';
    let u = vec![1.0; n];
    let a: Vec<f64> = (0..n).map(|i| 1.0 - tau.at(i)).collect();
    let neg_y: Vec<f64> = y.iter().map(|v| -v).collect();
    let b = x.t_mul_vec(&a);
    let sol = lp_fnm(x, &neg_y, &b, &u, &a, opt, method);
    RqFit { coef: sol.y.iter().map(|v| -v).collect(), iterations: sol.iterations, gap: sol.gap, converged: sol.converged }
}

/// Regression quantile(s) with rq.m's defaults: the coefficients of the tau-th quantile regression of y on X
/// (tau one number, or one per observation).
pub fn rq<'a>(x: &Matrix, y: &[f64], tau: impl Into<Tau<'a>>) -> Result<Vec<f64>, QrError> {
    rq_fit(x, y, tau, LpOptions::default(), Method::Qr).map(|f| f.coef)
}

/// rq with the normal equations solved by Cholesky (rq_fast in selection_qr_fast.py).
pub fn rq_fast<'a>(x: &Matrix, y: &[f64], tau: impl Into<Tau<'a>>) -> Result<Vec<f64>, QrError> {
    rq_fit(x, y, tau, LpOptions::default(), Method::Cholesky).map(|f| f.coef)
}

/// The check-function objective sum_i rho_tau(y_i - x_i'b) (tau one number or one per observation).
pub fn rq_objective<'a>(x: &Matrix, y: &[f64], b: &[f64], tau: impl Into<Tau<'a>>) -> f64 {
    let tau = tau.into();
    (0..x.nrow)
        .map(|i| {
            let e = y[i] - dot(x.row(i), b);
            e * (tau.at(i) - if e < 0.0 { 1.0 } else { 0.0 })
        })
        .sum()
}

// =================================================================================================
// rqic: quantile regression with linear inequality constraints R b >= r
// =================================================================================================

/// Solve min{ c1'x1 + c2'x2 | A1 x1 + A2 x2 = b, 0 < x1 < u, 0 < x2 } by the interior point method (lp_fnm in
/// rqic.m), with A1 = X1' and A2 = X2' given as the n1 x p and n2 x p matrices X1 and X2. The initial point
/// (x1, x2) must satisfy 0 < x1 < u and 0 < x2 but need not satisfy the equality constraints.
///
/// From Koenker's comments: based on the Lustig, Marsden and Shanno (1992) description of an implementation of
/// Mehrotra's (1990) predictor-corrector interior point method for bounded variable linear programming;
/// loosely based on the S-Plus and Fortran versions described in Portnoy and Koenker (1997).
#[allow(clippy::too_many_arguments)]
pub fn lp_fnm_ineq(
    x1m: &Matrix, c1: &[f64], x2m: &Matrix, c2: &[f64], b: &[f64], u: &[f64], x1_0: &[f64], x2_0: &[f64], opt: LpOptions,
) -> Result<LpSolution, QrError> {
    let (n1, n2, p) = (x1m.nrow, x2m.nrow, x1m.ncol);
    let (mut x1, mut x2) = (x1_0.to_vec(), x2_0.to_vec());
    // Generate an initial point
    let mut s: Vec<f64> = u.iter().zip(&x1).map(|(a, b)| a - b).collect();
    let mut y = Qr::new(x1m, None).solve(c1);
    let ya = x1m.mul_vec(&y);
    let mut z1: Vec<f64> = (0..n1).map(|i| (c1[i] - ya[i]).max(0.0)).collect();
    let mut w: Vec<f64> = (0..n1).map(|i| z1[i] - (c1[i] - ya[i])).collect();
    let mut z2 = vec![1.0; n2];
    let gap_of = |z1: &[f64], x1: &[f64], z2: &[f64], x2: &[f64], w: &[f64], s: &[f64]| dot(z1, x1) + dot(z2, x2) + dot(w, s);
    let mut gap = gap_of(&z1, &x1, &z2, &x2, &w, &s);

    let mut it = 0;
    while gap > opt.small && it < opt.max_it {
        it += 1;
        // Compute affine step
        let ya1 = x1m.mul_vec(&y);
        let ya2 = x2m.mul_vec(&y);
        let r1: Vec<f64> = (0..n1).map(|i| c1[i] - ya1[i]).collect();
        let r2: Vec<f64> = (0..n2).map(|i| c2[i] - ya2[i]).collect();
        let ax1 = x1m.t_mul_vec(&x1);
        let ax2 = x2m.t_mul_vec(&x2);
        let r3: Vec<f64> = (0..p).map(|j| b[j] - ax1[j] - ax2[j]).collect();
        let q1: Vec<f64> = (0..n1).map(|i| 1.0 / (z1[i] / x1[i] + w[i] / s[i])).collect();
        let q2: Vec<f64> = (0..n2).map(|i| x2[i] / z2[i]).collect();
        // AQA = AQ1 A1' + AQ2 A2'
        let mut aqa = x1m.weighted_gram(&q1);
        let g2 = x2m.weighted_gram(&q2);
        for (a, g) in aqa.data.iter_mut().zip(&g2.data) {
            *a += g;
        }
        let t1 = x1m.t_mul_vec(&q1.iter().zip(&r1).map(|(a, b)| a * b).collect::<Vec<_>>());
        let t2 = x2m.t_mul_vec(&q2.iter().zip(&r2).map(|(a, b)| a * b).collect::<Vec<_>>());
        let mut rhs: Vec<f64> = (0..p).map(|j| r3[j] + t1[j] + t2[j]).collect();
        let mut dy = lu_solve(&aqa, &rhs).ok_or_else(|| QrError::Singular("rqic: A Q A'".into()))?;
        let ady1 = x1m.mul_vec(&dy);
        let ady2 = x2m.mul_vec(&dy);
        let mut dx1: Vec<f64> = (0..n1).map(|i| q1[i] * (ady1[i] - r1[i])).collect();
        let mut dx2: Vec<f64> = (0..n2).map(|i| q2[i] * (ady2[i] - r2[i])).collect();
        let mut ds: Vec<f64> = dx1.iter().map(|v| -v).collect();
        let mut dz1: Vec<f64> = (0..n1).map(|i| -z1[i] * (1.0 + dx1[i] / x1[i])).collect();
        let mut dz2: Vec<f64> = (0..n2).map(|i| -z2[i] * (1.0 + dx2[i] / x2[i])).collect();
        let mut dw: Vec<f64> = (0..n1).map(|i| -w[i] * (1.0 + ds[i] / s[i])).collect();

        // Compute maximum allowable step lengths
        let mut fp = step_length(opt.beta, &[(&x1, &dx1), (&s, &ds), (&x2, &dx2)]);
        let mut fd = step_length(opt.beta, &[(&w, &dw), (&z1, &dz1), (&z2, &dz2)]);

        // If full affine step is feasible, take it. Otherwise modify it
        if fp.min(fd) < 1.0 {
            let mut mu = dot(&z1, &x1) + dot(&z2, &x2) + dot(&w, &s);
            let mut g = 0.0;
            for i in 0..n1 {
                g += (z1[i] + fd * dz1[i]) * (x1[i] + fp * dx1[i]);
            }
            for i in 0..n2 {
                g += (z2[i] + fd * dz2[i]) * (x2[i] + fp * dx2[i]);
            }
            for i in 0..n1 {
                g += (w[i] + fd * dw[i]) * (s[i] + fp * ds[i]);
            }
            // MATLAB: mu = mu * (g / mu) ^3 / ( 2* n1 + n2);
            mu = mu * (g / mu).powi(3) / (2.0 * n1 as f64 + n2 as f64);

            // Compute modified step
            let dxdz1: Vec<f64> = (0..n1).map(|i| dx1[i] * dz1[i]).collect();
            let dxdz2: Vec<f64> = (0..n2).map(|i| dx2[i] * dz2[i]).collect();
            let dsdw: Vec<f64> = (0..n1).map(|i| ds[i] * dw[i]).collect();
            let xinv1: Vec<f64> = x1.iter().map(|v| 1.0 / v).collect();
            let xinv2: Vec<f64> = x2.iter().map(|v| 1.0 / v).collect();
            let sinv: Vec<f64> = s.iter().map(|v| 1.0 / v).collect();
            let xi1: Vec<f64> = (0..n1).map(|i| xinv1[i] * dxdz1[i] - sinv[i] * dsdw[i] - mu * (xinv1[i] - sinv[i])).collect();
            let xi2: Vec<f64> = (0..n2).map(|i| xinv2[i] * dxdz2[i] - mu * xinv2[i]).collect();
            let a1 = x1m.t_mul_vec(&(0..n1).map(|i| q1[i] * xi1[i]).collect::<Vec<_>>());
            let a2 = x2m.t_mul_vec(&(0..n2).map(|i| q2[i] * xi2[i]).collect::<Vec<_>>());
            for j in 0..p {
                rhs[j] += a1[j] + a2[j];
            }
            dy = lu_solve(&aqa, &rhs).ok_or_else(|| QrError::Singular("rqic: A Q A'".into()))?;
            let ady1 = x1m.mul_vec(&dy);
            let ady2 = x2m.mul_vec(&dy);
            dx1 = (0..n1).map(|i| q1[i] * (ady1[i] - r1[i] - xi1[i])).collect();
            dx2 = (0..n2).map(|i| q2[i] * (ady2[i] - r2[i] - xi2[i])).collect();
            ds = dx1.iter().map(|v| -v).collect();
            dz1 = (0..n1).map(|i| -z1[i] + xinv1[i] * (mu - z1[i] * dx1[i] - dxdz1[i])).collect();
            dz2 = (0..n2).map(|i| -z2[i] + xinv2[i] * (mu - z2[i] * dx2[i] - dxdz2[i])).collect();
            dw = (0..n1).map(|i| -w[i] + sinv[i] * (mu - w[i] * ds[i] - dsdw[i])).collect();

            fp = step_length(opt.beta, &[(&x1, &dx1), (&s, &ds), (&x2, &dx2)]);
            fd = step_length(opt.beta, &[(&w, &dw), (&z1, &dz1), (&z2, &dz2)]);
        }

        // Take the step
        for i in 0..n1 {
            x1[i] += fp * dx1[i];
            z1[i] += fd * dz1[i];
            s[i] += fp * ds[i];
            w[i] += fd * dw[i];
        }
        for i in 0..n2 {
            x2[i] += fp * dx2[i];
            z2[i] += fd * dz2[i];
        }
        for (yj, d) in y.iter_mut().zip(&dy) {
            *yj += fd * d;
        }
        gap = gap_of(&z1, &x1, &z2, &x2, &w, &s);
    }
    Ok(LpSolution { converged: gap <= opt.small, y, iterations: it, gap })
}

/// Regression quantile subject to linear inequality constraints R b >= r (rqic.m), with rqic.m's defaults.
/// X is n x p (include a column of ones for an intercept); R is m x p and r has length m. Example: to force the
/// second coefficient to be nonnegative, R = [[0, 1, ...]], r = [0].
pub fn rqic(x: &Matrix, y: &[f64], r_mat: &Matrix, r: &[f64], tau: f64) -> Result<Vec<f64>, QrError> {
    rqic_fit(x, y, r_mat, r, tau, LpOptions::rqic_default()).map(|f| f.coef)
}

/// rqic with options and the solver's report.
pub fn rqic_fit(x: &Matrix, y: &[f64], r_mat: &Matrix, r: &[f64], tau: f64, opt: LpOptions) -> Result<RqFit, QrError> {
    let (n, p) = (x.nrow, x.ncol);
    let m = r_mat.nrow;
    if r_mat.ncol != p || r.len() != m {
        return Err(QrError::Shape("R should be m x p and r should have length m".into()));
    }
    if y.len() != n {
        return Err(QrError::Shape(format!("y should have length n = {n}")));
    }
    Tau::All(tau).check(n)?;
    // MATLAB: u = ones(n,1); a1 = (1-tau)*u; a2 = ones(m,1); b = X' * a1;
    //         beta = -lp_fnm(X', -y', R', -r', b, u, a1, a2)';
    let u = vec![1.0; n];
    let a1 = vec![1.0 - tau; n];
    let a2 = vec![1.0; m];
    let neg_y: Vec<f64> = y.iter().map(|v| -v).collect();
    let neg_r: Vec<f64> = r.iter().map(|v| -v).collect();
    let b = x.t_mul_vec(&a1);
    let sol = lp_fnm_ineq(x, &neg_y, r_mat, &neg_r, &b, &u, &a1, &a2, opt)?;
    Ok(RqFit { coef: sol.y.iter().map(|v| -v).collect(), iterations: sol.iterations, gap: sol.gap, converged: sol.converged })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_as_in_test_rq() {
        assert_eq!(bound(&[1.0, 2.0, 3.0], &[-0.5, 1.0, -3.0]), vec![2.0, 1e20, 1.0]);
    }

    #[test]
    fn median_of_a_constant() {
        let y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        let x = Matrix::from_row_major(9, 1, vec![1.0; 9]);
        let b = rq(&x, &y, 0.5).unwrap();
        assert!((b[0] - 4.0).abs() < 1e-3, "{b:?}");
    }

    #[test]
    fn input_checks() {
        let x = Matrix::from_row_major(3, 2, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(rq(&x, &[1.0, 2.0, 3.0], 0.5).unwrap_err(), QrError::RankDeficient);
        let x = Matrix::from_row_major(3, 1, vec![1.0; 3]);
        assert!(matches!(rq(&x, &[1.0, 2.0, 3.0], 1.0), Err(QrError::Tau(_))));
        assert!(matches!(rq(&x, &[1.0, 2.0], 0.5), Err(QrError::Shape(_))));
    }
}
