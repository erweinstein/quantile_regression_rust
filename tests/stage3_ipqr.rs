//! Stage 3: ipqr (Koenker and Park) against the Python port (ipqr.py), and against rq for the linear model.
mod common;
use common::*;
use quantile_regression::ipqr::{ipqr, FnModel, Linear};
use quantile_regression::linalg::Matrix;
use quantile_regression::rq::{rq, rq_objective};

#[test]
fn ipqr_linear_matches_python_and_rq() {
    let r = load("ipqr.json");
    let (x, y) = (mat(&r["X"]), vec(&r["y"]));
    for c in r["linear"].as_array().unwrap() {
        let tau = c["tau"].as_f64().unwrap();
        let f = ipqr(tau, &y, &Linear(&x), &[0.0; 3]).unwrap();
        let db = max_abs_diff(&f.beta, &vec(&c["beta"]));
        let lp = rq_objective(&x, &y, &rq(&x, &y, tau).unwrap(), tau);
        println!("linear tau {tau}: beta {db:.1e}; objective {} (Python {}, rq {lp}); iterations {} (Python {})",
                 f.objective, c["objective"], f.iterations, c["iterations"]);
        assert!(db < 1e-6, "{db:e}");
        assert_eq!(f.iterations as u64, c["iterations"].as_u64().unwrap());
        assert!((f.objective - lp).abs() < 1e-5);
    }
}

#[test]
fn ipqr_exponential_matches_python() {
    let r = load("ipqr.json");
    let (x, y) = (vec(&r["x_exp"]), vec(&r["y_exp"]));
    let model = FnModel {
        f: |b: &[f64]| x.iter().map(|xi| b[0] * (b[1] * xi).exp()).collect(),
        jac: |b: &[f64]| {
            let cols: Vec<Vec<f64>> = vec![x.iter().map(|xi| (b[1] * xi).exp()).collect(),
                                           x.iter().map(|xi| b[0] * xi * (b[1] * xi).exp()).collect()];
            Matrix::from_columns(&[&cols[0], &cols[1]])
        },
    };
    for c in r["exponential"].as_array().unwrap() {
        let tau = c["tau"].as_f64().unwrap();
        let f = ipqr(tau, &y, &model, &[1.0, 0.3]).unwrap();
        let db = max_abs_diff(&f.beta, &vec(&c["beta"]));
        println!("exponential tau {tau}: beta {:?} (diff {db:.1e}); objective {} (Python {}); iterations {} (Python {})",
                 f.beta, f.objective, c["objective"], f.iterations, c["iterations"]);
        assert!(db < 1e-6, "{db:e}");
        assert_eq!(f.iterations as u64, c["iterations"].as_u64().unwrap());
    }
}
