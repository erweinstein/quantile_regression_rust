//! Python bindings for the quantile_regression crate (PyO3). The functions take and return numpy arrays and
//! mirror the Python port (quantile_regression: rq.py, ipqr.py, selection_qr.py, selection_qr_fast.py,
//! qr_extensions.py, smoothed_qr.py): the same names, argument order, defaults and returned dicts, with these
//! differences: the bootstraps take an integer seed and draw from Xoshiro256++ streams (one per draw), so their
//! draws differ from numpy's; and the long computations release the GIL.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use quantile_regression as qr;
use qr::linalg::Matrix;

fn err(e: qr::rq::QrError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// a numpy array (2-D, or 1-D as one column) as a row-major Matrix
fn mat(a: &Bound<'_, PyAny>) -> PyResult<Matrix> {
    if let Ok(m) = a.extract::<PyReadonlyArray2<f64>>() {
        let v = m.as_array();
        let (r, c) = v.dim();
        return Ok(Matrix::from_row_major(r, c, v.iter().cloned().collect()));
    }
    let v: Vec<f64> = a.extract::<PyReadonlyArray1<f64>>().map(|x| x.as_array().to_vec()).or_else(|_| a.extract::<Vec<f64>>())?;
    Ok(Matrix::from_row_major(v.len(), 1, v))
}

fn vec1(a: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(x) = a.extract::<PyReadonlyArray1<f64>>() {
        return Ok(x.as_array().to_vec());
    }
    if let Ok(x) = a.extract::<PyReadonlyArray1<i64>>() {
        return Ok(x.as_array().iter().map(|&v| v as f64).collect());
    }
    if let Ok(x) = a.extract::<PyReadonlyArray1<bool>>() {
        return Ok(x.as_array().iter().map(|&v| if v { 1.0 } else { 0.0 }).collect());
    }
    a.extract::<Vec<f64>>()
}

fn opt_vec(a: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<f64>>> {
    match a {
        Some(x) if !x.is_none() => Ok(Some(vec1(x)?)),
        _ => Ok(None),
    }
}

fn to_py2<'py>(py: Python<'py>, m: &Matrix) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((m.nrow, m.ncol), m.data.clone()).unwrap().into_pyarray(py)
}

fn to_py1<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, v)
}

enum TauIn {
    All(f64),
    Per(Vec<f64>),
}

fn tau_in(t: &Bound<'_, PyAny>) -> PyResult<TauIn> {
    if let Ok(v) = t.extract::<f64>() {
        return Ok(TauIn::All(v));
    }
    Ok(TauIn::Per(vec1(t)?))
}

fn method(m: &str) -> PyResult<qr::rq::Method> {
    match m {
        "qr" => Ok(qr::rq::Method::Qr),
        "cholesky" => Ok(qr::rq::Method::Cholesky),
        _ => Err(PyValueError::new_err("method should be 'qr' or 'cholesky'")),
    }
}

/// rq(X, y, tau=0.5, beta=0.9995, small=1e-5, max_it=50, method="qr"): linear quantile regression by the
/// Frisch-Newton interior point method (Koenker, Morillo and Eilers' rq.m). tau is a number or one value per
/// observation. method="cholesky" solves the Newton systems by Cholesky (rq_fast in selection_qr_fast.py).
#[pyfunction]
#[pyo3(signature = (x, y, tau=None, beta=0.9995, small=1e-5, max_it=50, method="qr"))]
fn rq<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, tau: Option<&Bound<'py, PyAny>>, beta: f64, small: f64,
           max_it: usize, method: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (x, y) = (mat(x)?, vec1(y)?);
    let t = match tau { Some(t) => tau_in(t)?, None => TauIn::All(0.5) };
    let m = self::method(method)?;
    let opt = qr::rq::LpOptions { beta, small, max_it };
    let b = py.detach(|| match &t {
        TauIn::All(v) => qr::rq::rq_fit(&x, &y, *v, opt, m),
        TauIn::Per(v) => qr::rq::rq_fit(&x, &y, v, opt, m),
    }).map_err(err)?;
    Ok(to_py1(py, b.coef))
}

/// rq_objective(X, y, b, tau): the check-function objective sum_i rho_tau(y_i - x_i'b).
#[pyfunction]
fn rq_objective(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, tau: &Bound<'_, PyAny>) -> PyResult<f64> {
    let (x, y, b) = (mat(x)?, vec1(y)?, vec1(b)?);
    Ok(match tau_in(tau)? {
        TauIn::All(v) => qr::rq::rq_objective(&x, &y, &b, v),
        TauIn::Per(v) => qr::rq::rq_objective(&x, &y, &b, &v),
    })
}

/// rqic(X, y, R, r, tau=0.5): quantile regression subject to the linear inequality constraints R b >= r
/// (Koenker, Morillo and Eilers' rqic.m).
#[pyfunction]
#[pyo3(signature = (x, y, r_mat, r, tau=0.5))]
fn rqic<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, r_mat: &Bound<'py, PyAny>, r: &Bound<'py, PyAny>,
             tau: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (x, y, rm, rv) = (mat(x)?, vec1(y)?, mat(r_mat)?, vec1(r)?);
    let b = py.detach(|| qr::rq::rqic(&x, &y, &rm, &rv, tau)).map_err(err)?;
    Ok(to_py1(py, b))
}

/// ipqr(q, y, x, func=None, dfunc=None, beta=None, toler=1e-8, eta=0.97, inner=2, max_iter=500): nonlinear
/// regression quantile by Koenker and Park's interior point algorithm (ipqr.m). With func and dfunc None the
/// model is linear, f(x, b) = x @ b (x an n x p array, run entirely in Rust); otherwise func(x, beta) -> (n,)
/// and dfunc(x, beta) -> (n, p) are Python functions, called from Rust at every iteration. Returns a dict with
/// beta, objective and iterations.
#[pyfunction]
#[pyo3(signature = (q, y, x, func=None, dfunc=None, beta=None, toler=1e-8, eta=0.97, inner=2, max_iter=500))]
#[allow(clippy::too_many_arguments)]
fn ipqr<'py>(py: Python<'py>, q: f64, y: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>, func: Option<Bound<'py, PyAny>>,
             dfunc: Option<Bound<'py, PyAny>>, beta: Option<&Bound<'py, PyAny>>, toler: f64, eta: f64, inner: usize,
             max_iter: usize) -> PyResult<Bound<'py, PyDict>> {
    let y = vec1(y)?;
    let b0 = beta.map(vec1).transpose()?.ok_or_else(|| PyValueError::new_err("beta (the starting value) is required"))?;
    let opt = qr::ipqr::IpqrOptions { toler, eta, inner, max_iter };
    let fit = match (func, dfunc) {
        (None, None) => {
            let xm = mat(x)?;
            py.detach(|| qr::ipqr::ipqr_with(q, &y, &qr::ipqr::Linear(&xm), &b0, opt)).map_err(err)?
        }
        (Some(f), Some(df)) => {
            let xo = x.clone().unbind();
            let (f, df) = (f.unbind(), df.unbind());
            let failed: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
            let call_v = |b: &[f64]| -> Vec<f64> {
                let r = f.bind(py).call1((xo.bind(py), PyArray1::from_slice(py, b))).and_then(|v| vec1(&v));
                r.unwrap_or_else(|e| { failed.borrow_mut().get_or_insert(e); vec![f64::NAN; y.len()] })
            };
            let call_j = |b: &[f64]| -> Matrix {
                let r = df.bind(py).call1((xo.bind(py), PyArray1::from_slice(py, b))).and_then(|v| mat(&v));
                r.unwrap_or_else(|e| { failed.borrow_mut().get_or_insert(e); Matrix::zeros(y.len(), b.len()) })
            };
            let model = qr::ipqr::FnModel { f: call_v, jac: call_j };
            let out = qr::ipqr::ipqr_with(q, &y, &model, &b0, opt);
            if let Some(e) = failed.into_inner() { return Err(e); }
            out.map_err(err)?
        }
        _ => return Err(PyValueError::new_err("give both func and dfunc, or neither (linear model)")),
    };
    let d = PyDict::new(py);
    d.set_item("beta", to_py1(py, fit.beta))?;
    d.set_item("objective", fit.objective)?;
    d.set_item("iterations", fit.iterations)?;
    Ok(d)
}

/// probit(D, Z, w=None): probit coefficients (constant first) by Fisher scoring, optionally weighted.
#[pyfunction]
#[pyo3(signature = (d, z, w=None))]
fn probit<'py>(py: Python<'py>, d: &Bound<'py, PyAny>, z: &Bound<'py, PyAny>, w: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (d, z, w) = (vec1(d)?, mat(z)?, opt_vec(w)?);
    Ok(to_py1(py, qr::selection::probit(&d, &z, w.as_deref()).map_err(err)?))
}

/// bivariate_normal_cdf(h, k, rho): P(X <= h, Y <= k) for a standard bivariate normal with correlation rho.
#[pyfunction]
fn bivariate_normal_cdf(h: f64, k: f64, rho: f64) -> f64 {
    qr::normal::bvn_cdf(h, k, rho)
}

/// rotated_tau(tau, p, rho): the rotated quantile levels G_tau(p; rho) of Arellano and Bonhomme (2017).
#[pyfunction]
fn rotated_tau<'py>(py: Python<'py>, tau: f64, p: &Bound<'py, PyAny>, rho: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(to_py1(py, qr::selection::rotated_tau(tau, &vec1(p)?, rho)))
}

fn sel_opts(vectau: Option<&Bound<'_, PyAny>>, vecrho: Option<&Bound<'_, PyAny>>, taus: Option<&Bound<'_, PyAny>>) -> PyResult<qr::selection::SelectionOptions> {
    let mut o = qr::selection::SelectionOptions::default();
    if let Some(v) = opt_vec(vectau)? { o.vectau = v; }
    if let Some(v) = opt_vec(vecrho)? { o.vecrho = v; }
    if let Some(v) = opt_vec(taus)? { o.taus = v; }
    Ok(o)
}

fn sel_dict<'py>(py: Python<'py>, f: &qr::selection::SelectionFit) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("rho", f.rho)?;
    d.set_item("beta", to_py2(py, &f.beta))?;
    d.set_item("taus", to_py1(py, f.taus.clone()))?;
    d.set_item("objective", to_py1(py, f.objective.clone()))?;
    d.set_item("vecrho", to_py1(py, f.vecrho.clone()))?;
    d.set_item("gamma", to_py1(py, f.gamma.clone()))?;
    d.set_item("pZ", to_py1(py, f.pz.clone()))?;
    Ok(d)
}

/// selection_qr(D, Y, X, B, vectau=None, vecrho=None, taus=None): Arellano and Bonhomme's (2017) quantile
/// regression with sample selection, as Code_QRS.m (defaults: vectau .2, .4, .6, .8; the 99-point rho grid;
/// taus .05, ..., .95). Returns the dict of selection_qr.py: rho, beta, taus, objective, vecrho, gamma, pZ.
#[pyfunction]
#[pyo3(signature = (d, y, x, b, vectau=None, vecrho=None, taus=None))]
#[allow(clippy::too_many_arguments)]
fn selection_qr<'py>(py: Python<'py>, d: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>,
                     vectau: Option<&Bound<'py, PyAny>>, vecrho: Option<&Bound<'py, PyAny>>, taus: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyDict>> {
    let (d, y, x, b) = (vec1(d)?, vec1(y)?, mat(x)?, mat(b)?);
    let o = sel_opts(vectau, vecrho, taus)?;
    let f = py.detach(|| qr::selection::selection_qr(&d, &y, &x, &b, &o)).map_err(err)?;
    sel_dict(py, &f)
}

/// selection_qr_weighted(D, Y, X, B, w=None, vectau=None, vecrho=None, taus=None): selection_qr with
/// observation weights in the probit, the quantile regressions and the moment conditions.
#[pyfunction]
#[pyo3(signature = (d, y, x, b, w=None, vectau=None, vecrho=None, taus=None))]
#[allow(clippy::too_many_arguments)]
fn selection_qr_weighted<'py>(py: Python<'py>, d: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>,
                              w: Option<&Bound<'py, PyAny>>, vectau: Option<&Bound<'py, PyAny>>, vecrho: Option<&Bound<'py, PyAny>>,
                              taus: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyDict>> {
    let (d, y, x, b, w) = (vec1(d)?, vec1(y)?, mat(x)?, mat(b)?, opt_vec(w)?);
    let o = sel_opts(vectau, vecrho, taus)?;
    let f = py.detach(|| qr::selection::selection_qr_weighted(&d, &y, &x, &b, w.as_deref(), &o)).map_err(err)?;
    sel_dict(py, &f)
}

/// selection_qr_bootstrap(D, Y, X, B, reps=200, seed=0, vectau=None, vecrho=None, taus=None): the weighted
/// (Exp(1) multiplier) bootstrap of selection_qr_weighted, the draws run in parallel (RAYON_NUM_THREADS sets
/// the threads; the results do not depend on it). Returns estimate, rho_draws, beta_draws (reps x (1 + dim X)
/// x len(taus)), rho_se and beta_se.
#[pyfunction]
#[pyo3(signature = (d, y, x, b, reps=200, seed=0, vectau=None, vecrho=None, taus=None))]
#[allow(clippy::too_many_arguments)]
fn selection_qr_bootstrap<'py>(py: Python<'py>, d: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>,
                               reps: usize, seed: u64, vectau: Option<&Bound<'py, PyAny>>, vecrho: Option<&Bound<'py, PyAny>>,
                               taus: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyDict>> {
    let (d, y, x, b) = (vec1(d)?, vec1(y)?, mat(x)?, mat(b)?);
    let o = sel_opts(vectau, vecrho, taus)?;
    let r = py.detach(|| qr::selection::selection_qr_bootstrap(&d, &y, &x, &b, reps, seed, &o)).map_err(err)?;
    let (p, t) = (r.beta_se.nrow, r.beta_se.ncol);
    let draws = Array3::from_shape_vec((r.beta_draws.len(), p, t), r.beta_draws.iter().flat_map(|m| m.data.iter().cloned()).collect()).unwrap();
    let out = PyDict::new(py);
    out.set_item("estimate", sel_dict(py, &r.estimate)?)?;
    out.set_item("rho_draws", to_py1(py, r.rho_draws.clone()))?;
    out.set_item("beta_draws", draws.into_pyarray(py))?;
    out.set_item("rho_se", r.rho_se)?;
    out.set_item("beta_se", to_py2(py, &r.beta_se))?;
    Ok(out)
}

/// selection_qr_fast(D, Y, X, B, settings="code", vectau=None, vecrho=None, taus=None, preprocess=False,
/// search="full", tie_tol=0.0, coarse_step=5, n_best=3): the faster computation of selection_qr
/// (selection_qr_fast.py): settings "code" (Code_QRS.m's grids) or "paper"; preprocess = Portnoy and Koenker's
/// preprocessing; search "full" or "coarse". Returns the selection_qr dict plus `evaluated` (grid points
/// evaluated; unevaluated objectives are NaN).
#[pyfunction]
#[pyo3(signature = (d, y, x, b, settings="code", vectau=None, vecrho=None, taus=None, preprocess=false, search="full", tie_tol=0.0, coarse_step=5, n_best=3))]
#[allow(clippy::too_many_arguments)]
fn selection_qr_fast<'py>(py: Python<'py>, d: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>,
                          settings: &str, vectau: Option<&Bound<'py, PyAny>>, vecrho: Option<&Bound<'py, PyAny>>, taus: Option<&Bound<'py, PyAny>>,
                          preprocess: bool, search: &str, tie_tol: f64, coarse_step: usize, n_best: usize) -> PyResult<Bound<'py, PyDict>> {
    let (d, y, x, b) = (vec1(d)?, vec1(y)?, mat(x)?, mat(b)?);
    let mut o = qr::fast::FastOptions::default();
    o.settings = match settings { "code" => qr::fast::Settings::Code, "paper" => qr::fast::Settings::Paper,
        _ => return Err(PyValueError::new_err("settings should be 'code' or 'paper'")) };
    o.vectau = opt_vec(vectau)?; o.vecrho = opt_vec(vecrho)?;
    if let Some(t) = opt_vec(taus)? { o.taus = t; }
    o.preprocess = preprocess; o.tie_tol = tie_tol;
    o.search = match search { "full" => qr::fast::Search::Full, "coarse" => qr::fast::Search::Coarse { coarse_step, n_best },
        _ => return Err(PyValueError::new_err("search should be 'full' or 'coarse'")) };
    let f = py.detach(|| qr::fast::selection_qr_fast(&d, &y, &x, &b, &o)).map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("rho", f.rho)?;
    out.set_item("beta", to_py2(py, &f.beta))?;
    out.set_item("taus", to_py1(py, f.taus))?;
    out.set_item("objective", to_py1(py, f.objective))?;
    out.set_item("vecrho", to_py1(py, f.vecrho))?;
    out.set_item("gamma", to_py1(py, f.gamma))?;
    out.set_item("pZ", to_py1(py, f.pz))?;
    out.set_item("evaluated", f.evaluated)?;
    Ok(out)
}

/// hall_sheather(n, tau, alpha=0.05): the Hall-Sheather bandwidth.
#[pyfunction]
#[pyo3(signature = (n, tau, alpha=0.05))]
fn hall_sheather(n: usize, tau: f64, alpha: f64) -> f64 {
    qr::extensions::hall_sheather(n, tau, alpha)
}

/// rq_se(X, y, tau=0.5, beta=None, method="ker", alpha=0.05): standard errors of rq(X, y, tau) ("iid" or
/// "ker") and their covariance matrix, as (se, cov).
#[pyfunction]
#[pyo3(signature = (x, y, tau=0.5, beta=None, method="ker", alpha=0.05))]
fn rq_se<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, tau: f64, beta: Option<&Bound<'py, PyAny>>, method: &str,
              alpha: f64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (x, y, b) = (mat(x)?, vec1(y)?, opt_vec(beta)?);
    let m = match method { "iid" => qr::extensions::SeMethod::Iid, "ker" => qr::extensions::SeMethod::Ker,
        _ => return Err(PyValueError::new_err("method should be 'iid' or 'ker'")) };
    let (se, cov) = qr::extensions::rq_se(&x, &y, tau, b.as_deref(), m, alpha).map_err(err)?;
    Ok((to_py1(py, se), to_py2(py, &cov)))
}

/// rearrange(beta, x): monotone conditional quantile curves x'beta(tau), sorted over tau (beta: p x J; x: m x p
/// including the leading 1). Returns m x J.
#[pyfunction]
fn rearrange<'py>(py: Python<'py>, beta: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Ok(to_py2(py, &qr::extensions::rearrange(&mat(beta)?, &mat(x)?)))
}

/// default_bandwidth(n, p): conquer's default bandwidth.
#[pyfunction]
fn default_bandwidth(n: usize, p: usize) -> f64 {
    qr::smoothed::default_bandwidth(n, p)
}

/// conquer(X, y, tau=0.5, h=None, w=None, b0=None, tol=1e-8, max_iter=100): convolution-smoothed quantile
/// regression (He, Pan, Tan and Zhou 2023), Gaussian kernel. X includes the constant column if wanted.
#[pyfunction]
#[pyo3(signature = (x, y, tau=0.5, h=None, w=None, b0=None, tol=1e-8, max_iter=100))]
#[allow(clippy::too_many_arguments)]
fn conquer<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, tau: f64, h: Option<f64>, w: Option<&Bound<'py, PyAny>>,
                b0: Option<&Bound<'py, PyAny>>, tol: f64, max_iter: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (x, y, w, b0) = (mat(x)?, vec1(y)?, opt_vec(w)?, opt_vec(b0)?);
    let b = py.detach(|| qr::smoothed::conquer(&x, &y, tau, h, w.as_deref(), b0.as_deref(), tol, max_iter));
    Ok(to_py1(py, b))
}

/// conquer_bootstrap(X, y, tau=0.5, h=None, reps=1000, alpha=0.05, seed=0): the Exp(1) multiplier bootstrap of
/// conquer, in parallel. Returns estimate, draws, se, h, and the percentile, pivotal and normal intervals.
#[pyfunction]
#[pyo3(signature = (x, y, tau=0.5, h=None, reps=1000, alpha=0.05, seed=0))]
#[allow(clippy::too_many_arguments)]
fn conquer_bootstrap<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, tau: f64, h: Option<f64>, reps: usize, alpha: f64,
                          seed: u64) -> PyResult<Bound<'py, PyDict>> {
    let (x, y) = (mat(x)?, vec1(y)?);
    let r = py.detach(|| qr::smoothed::conquer_bootstrap(&x, &y, tau, h, reps, alpha, seed));
    let p = r.estimate.len();
    let pairs = |v: &[(f64, f64)]| Array2::from_shape_vec((v.len(), 2), v.iter().flat_map(|&(a, b)| [a, b]).collect()).unwrap();
    let out = PyDict::new(py);
    out.set_item("estimate", to_py1(py, r.estimate.clone()))?;
    out.set_item("draws", Array2::from_shape_vec((r.draws.len(), p), r.draws.concat()).unwrap().into_pyarray(py))?;
    out.set_item("se", to_py1(py, r.se.clone()))?;
    out.set_item("h", r.h)?;
    out.set_item("percentile", pairs(&r.percentile).into_pyarray(py))?;
    out.set_item("pivotal", pairs(&r.pivotal).into_pyarray(py))?;
    out.set_item("normal", pairs(&r.normal).into_pyarray(py))?;
    Ok(out)
}

#[pymodule]
fn quantile_regression_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rq, m)?)?;
    m.add_function(wrap_pyfunction!(rq_objective, m)?)?;
    m.add_function(wrap_pyfunction!(rqic, m)?)?;
    m.add_function(wrap_pyfunction!(ipqr, m)?)?;
    m.add_function(wrap_pyfunction!(probit, m)?)?;
    m.add_function(wrap_pyfunction!(bivariate_normal_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(rotated_tau, m)?)?;
    m.add_function(wrap_pyfunction!(selection_qr, m)?)?;
    m.add_function(wrap_pyfunction!(selection_qr_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(selection_qr_bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(selection_qr_fast, m)?)?;
    m.add_function(wrap_pyfunction!(hall_sheather, m)?)?;
    m.add_function(wrap_pyfunction!(rq_se, m)?)?;
    m.add_function(wrap_pyfunction!(rearrange, m)?)?;
    m.add_function(wrap_pyfunction!(default_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(conquer, m)?)?;
    m.add_function(wrap_pyfunction!(conquer_bootstrap, m)?)?;
    Ok(())
}
