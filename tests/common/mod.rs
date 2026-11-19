//! Helpers for the tests: load the reference outputs written by reference/make_reference.py from the Python
//! code in erweinstein/quantile_regression.
#![allow(dead_code)]
use quantile_regression::linalg::Matrix;
use serde_json::Value;

pub fn load(name: &str) -> Value {
    let path = format!("{}/tests/reference/{}", env!("CARGO_MANIFEST_DIR"), name);
    serde_json::from_str(&std::fs::read_to_string(&path).expect("reference file")).expect("valid JSON")
}
pub fn vec(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}
pub fn mat(v: &Value) -> Matrix {
    let rows: Vec<Vec<f64>> = v.as_array().expect("array").iter().map(vec).collect();
    let (n, p) = (rows.len(), rows[0].len());
    Matrix::from_row_major(n, p, rows.concat())
}
/// A 1-D array as an n x 1 matrix.
pub fn col(v: &Value) -> Matrix {
    let x = vec(v);
    Matrix::from_row_major(x.len(), 1, x)
}
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
