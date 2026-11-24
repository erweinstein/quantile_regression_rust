//! multicore.rs -- does anything depend on the number of threads, and how does the parallel work scale?
//!
//!   multicore            run four parallel computations and print, for each, its time and a digest of every
//!                        bit of its result (run it with RAYON_NUM_THREADS = 1, 2, 3, ...: the digests must agree)
//!   multicore --tasks    time each independent piece of selection_qr on one thread (the ρ-grid evaluations and
//!                        the final quantiles, and the serial setup), for checks/scaling_sim.py
//!
//! The data are simulated here (the design of checks/benchmark.py), so the program needs no files.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use quantile_regression::fast::{selection_qr_fast, FastOptions};
use quantile_regression::linalg::{dot, Matrix};
use quantile_regression::normal;
use quantile_regression::rq::{rq_unchecked, LpOptions, Method, Tau};
use quantile_regression::selection::{probit, rotated_tau_q, selection_qr, selection_qr_bootstrap, SelectionOptions};
use quantile_regression::smoothed::conquer_bootstrap;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;

/// FNV-1a over the bits of the values: equal digests mean bit-for-bit equal results.
fn digest<'a>(vals: impl IntoIterator<Item = &'a f64>) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in vals {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}

/// The selection design of checks/benchmark.py: X, B ~ N(0, 1); (u, v) bivariate normal with correlation 0.5;
/// D = 1{0.3 + 0.5 X + B + v > 0}; Y = 1 + X + (0.5 + 0.3 X) u.
fn selection_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Matrix, Matrix) {
    let mut r = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut z = || -> f64 { StandardNormal.sample(&mut r) };
    let (mut d, mut y, mut x, mut b) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let (xi, bi, e1, e2) = (z(), z(), z(), z());
        let (u, v) = (e1, 0.5 * e1 + (0.75f64).sqrt() * e2);
        d.push(if 0.3 + 0.5 * xi + bi + v > 0.0 { 1.0 } else { 0.0 });
        y.push(1.0 + xi + (0.5 + 0.3 * xi) * u);
        x.push(xi);
        b.push(bi);
    }
    (d, y, Matrix::from_row_major(n, 1, x), Matrix::from_row_major(n, 1, b))
}

fn timed<T>(f: impl FnOnce() -> T) -> (f64, T) {
    let t = Instant::now();
    let r = f();
    (t.elapsed().as_secs_f64(), r)
}

fn main() {
    let threads = rayon::current_num_threads();
    if std::env::args().any(|a| a == "--tasks") {
        tasks();
        return;
    }
    let opt = SelectionOptions::default();
    let (d, y, x, b) = selection_data(8000, 1);
    let (t, f) = timed(|| selection_qr(&d, &y, &x, &b, &opt).unwrap());
    println!("threads={threads} task=selection_qr_n8000 secs={t:.3} rho={} digest={}", f.rho, digest(f.beta.data.iter().chain(&f.objective)));
    let (t, f) = timed(|| selection_qr_fast(&d, &y, &x, &b, &FastOptions { preprocess: true, ..Default::default() }).unwrap());
    println!("threads={threads} task=selection_qr_fast_n8000 secs={t:.3} rho={} digest={}", f.rho, digest(f.beta.data.iter()));
    let (d2, y2, x2, b2) = selection_data(2000, 2);
    let (t, bs) = timed(|| selection_qr_bootstrap(&d2, &y2, &x2, &b2, 16, 2017, &opt).unwrap());
    println!("threads={threads} task=bootstrap_16_n2000 secs={t:.3} rho_se={:.5} digest={}", bs.rho_se,
             digest(bs.rho_draws.iter().chain(&bs.beta_se.data)));
    // conquer: X = [1, x], y = 1 + x + t(4)-like errors from the same generator
    let mut r = Xoshiro256PlusPlus::seed_from_u64(3);
    let xs: Vec<f64> = (0..5000).map(|_| StandardNormal.sample(&mut r)).collect();
    let ys: Vec<f64> = xs.iter().map(|v| { let e: f64 = StandardNormal.sample(&mut r); 1.0 + v + e * (1.0 + 0.5 * v.abs()) }).collect();
    let xm = Matrix::from_columns(&[&vec![1.0; 5000], &xs]);
    let (t, cb) = timed(|| conquer_bootstrap(&xm, &ys, 0.5, None, 400, 0.05, 7));
    println!("threads={threads} task=conquer_bootstrap_400_n5000 secs={t:.3} se={:.5} digest={}", cb.se[1],
             digest(cb.draws.iter().flatten()));
}

/// One thread: the serial setup of selection_qr, then each independent task, as selection.rs runs them.
fn tasks() {
    for (n, seed) in [(2000usize, 9000u64), (8000, 1)] {
        let opt = SelectionOptions::default();
        let (d, y, x, b) = selection_data(n, seed);
        let (t_setup, (x1c, y1, pz1, q1)) = timed(|| {
            let z = x.hstack(&b);
            let g = probit(&d, &z, None).unwrap();
            let pz: Vec<f64> = (0..n).map(|i| normal::cdf(g[0] + dot(z.row(i), &g[1..]))).collect();
            let sel: Vec<usize> = (0..n).filter(|&i| d[i] == 1.0).collect();
            let x1c = x.select_rows(&sel).with_intercept();
            let y1: Vec<f64> = sel.iter().map(|&i| y[i]).collect();
            let pz1: Vec<f64> = sel.iter().map(|&i| pz[i]).collect();
            let q1: Vec<f64> = pz1.iter().map(|&p| normal::ppf(p)).collect();
            (x1c, y1, pz1, q1)
        });
        println!("n={n} kind=serial secs={t_setup:.6}");
        let fit = |tau: f64, rho: f64| {
            let g = rotated_tau_q(tau, &pz1, &q1, rho);
            (rq_unchecked(&x1c, &y1, Tau::PerObs(&g), LpOptions::default(), Method::Qr).coef, g)
        };
        for &rho in &opt.vecrho {
            let (t, _) = timed(|| {
                let mut obj = 0.0;
                for &tau in &opt.vectau {
                    let (beta, g) = fit(tau, rho);
                    let m: f64 = (0..y1.len()).map(|i| pz1[i] * (if y1[i] <= dot(x1c.row(i), &beta) { 1.0 } else { 0.0 } - g[i])).sum();
                    obj += m / y1.len() as f64;
                }
                obj * obj
            });
            println!("n={n} kind=grid rho={rho} secs={t:.6}");
        }
        for &tau in &opt.taus {
            let (t, _) = timed(|| fit(tau, 0.0));
            println!("n={n} kind=final tau={tau} secs={t:.6}");
        }
    }
}
