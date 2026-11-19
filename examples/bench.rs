//! bench.rs -- compute-only timings for checks/benchmark.py: reads a CSV written by that script and times one
//! estimator, printing "seconds,result". Usage: bench (rq|rq_chol|sel|sel_fast|ipqr) DATA.csv [tau]
use quantile_regression::fast::{selection_qr_fast, FastOptions};
use quantile_regression::ipqr::{ipqr, FnModel};
use quantile_regression::linalg::Matrix;
use quantile_regression::rq::{rq_fit, LpOptions, Method};
use quantile_regression::selection::{selection_qr, SelectionOptions};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&a[2]).unwrap();
    let rows: Vec<Vec<f64>> = text.lines().skip(1).map(|l| l.split(',').map(|s| s.parse().unwrap_or(f64::NAN)).collect()).collect();
    let c = |j: usize| -> Vec<f64> { rows.iter().map(|r| r[j]).collect() };
    let tau: f64 = a.get(3).map_or(0.5, |s| s.parse().unwrap());
    let t = Instant::now();
    let res = match a[1].as_str() {
        "rq" | "rq_chol" => {
            // columns y, x1, x2: X = [1, x1, x2]
            let x = Matrix::from_columns(&[&vec![1.0; rows.len()], &c(1), &c(2)]);
            let m = if a[1] == "rq" { Method::Qr } else { Method::Cholesky };
            format!("{:?}", rq_fit(&x, &c(0), tau, LpOptions::default(), m).unwrap().coef)
        }
        "sel" | "sel_fast" => {
            let (d, y, x, b) = (c(0), c(1), Matrix::from_columns(&[&c(2)]), Matrix::from_columns(&[&c(3)]));
            if a[1] == "sel" {
                format!("{}", selection_qr(&d, &y, &x, &b, &SelectionOptions::default()).unwrap().rho)
            } else {
                format!("{}", selection_qr_fast(&d, &y, &x, &b, &FastOptions { preprocess: true, ..Default::default() }).unwrap().rho)
            }
        }
        "ipqr" => {
            let (x, y) = (c(0), c(1));
            let m = FnModel {
                f: |b: &[f64]| x.iter().map(|xi| b[0] * (b[1] * xi).exp()).collect(),
                jac: |b: &[f64]| {
                    let (u, v): (Vec<f64>, Vec<f64>) = x.iter().map(|xi| ((b[1] * xi).exp(), b[0] * xi * (b[1] * xi).exp())).unzip();
                    Matrix::from_columns(&[&u, &v])
                },
            };
            format!("{:?}", ipqr(tau, &y, &m, &[1.0, 0.3]).unwrap().beta)
        }
        k => panic!("unknown kind {k}"),
    };
    println!("{:.6},{}", t.elapsed().as_secs_f64(), res.replace(',', ";"));
}
