//! qrs -- selection-corrected quantile regression (Arellano and Bonhomme 2017) on a CSV file, as Code_QRS.m
//! is run on a data set.
//!
//! Usage:
//!   qrs DATA.csv --d D --y Y --x X1,X2 --b B1 [--paper] [--fast] [--bootstrap REPS] [--seed SEED] [--out OUT.csv]
//!
//! DATA.csv has a header row; D is the participation indicator (1 = participant), Y the outcome (empty or NA
//! for nonparticipants is fine), X the covariates and B the excluded covariates (comma-separated column names;
//! no constant). Writes beta(tau) for tau = .05, ..., .95 (a row per tau; with --bootstrap, also weighted
//! bootstrap standard errors) to OUT.csv (default: standard output) and prints rho and the probit on stderr.
//! --paper uses the paper's copula search (the nine deciles and 200 grid points) with --fast's warm starts;
//! --fast alone uses Code_QRS.m's grids with warm starts and preprocessing (the same optima; rho can differ by
//! a grid step, see fast.rs). The default is Code_QRS.m exactly.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use quantile_regression::fast::{selection_qr_fast, FastOptions, Settings};
use quantile_regression::linalg::Matrix;
use quantile_regression::selection::{selection_qr, selection_qr_bootstrap, SelectionOptions};
use std::io::Write;

fn die(msg: &str) -> ! {
    eprintln!("qrs: {msg}");
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: qrs DATA.csv --d D --y Y --x X1,X2 --b B1 [--paper] [--fast] [--bootstrap REPS] [--seed SEED] [--out OUT.csv]");
        std::process::exit(if args.is_empty() { 2 } else { 0 });
    }
    let get = |flag: &str| args.iter().position(|a| a == flag).map(|i| args.get(i + 1).cloned().unwrap_or_else(|| die(&format!("{flag} needs a value"))));
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let text = std::fs::read_to_string(&args[0]).unwrap_or_else(|e| die(&format!("cannot read {}: {e}", args[0])));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines.next().unwrap_or_else(|| die("empty file")).split(',').map(|s| s.trim().trim_matches('"').to_string()).collect();
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split(',').map(|s| { let s = s.trim().trim_matches('"'); if s.is_empty() || s == "NA" || s == "NaN" { f64::NAN } else { s.parse().unwrap_or_else(|_| die(&format!("not a number: {s}"))) } }).collect())
        .collect();
    let col = |name: &str| -> Vec<f64> {
        let j = header.iter().position(|h| h == name).unwrap_or_else(|| die(&format!("no column {name}")));
        rows.iter().map(|r| r[j]).collect()
    };
    let cols = |spec: &str| -> Matrix {
        let names: Vec<&str> = spec.split(',').collect();
        let c: Vec<Vec<f64>> = names.iter().map(|n| col(n)).collect();
        Matrix::from_columns(&c.iter().map(|v| v.as_slice()).collect::<Vec<_>>())
    };
    let d = col(&get("--d").unwrap_or_else(|| die("--d is required")));
    let y = col(&get("--y").unwrap_or_else(|| die("--y is required")));
    let xs = get("--x").unwrap_or_else(|| die("--x is required"));
    let x = cols(&xs);
    let b = cols(&get("--b").unwrap_or_else(|| die("--b is required")));
    // complete cases in D, X and B (Y may be missing for nonparticipants)
    let ok: Vec<usize> = (0..d.len())
        .filter(|&i| !d[i].is_nan() && x.row(i).iter().chain(b.row(i)).all(|v| !v.is_nan()) && (d[i] != 1.0 || !y[i].is_nan()))
        .collect();
    if ok.len() < d.len() {
        eprintln!("qrs: dropped {} rows with missing values", d.len() - ok.len());
    }
    let (d, y, x, b) = (ok.iter().map(|&i| d[i]).collect::<Vec<_>>(), ok.iter().map(|&i| y[i]).collect::<Vec<_>>(), x.select_rows(&ok), b.select_rows(&ok));

    let (rho, beta, taus, gamma, se) = if has("--paper") || has("--fast") {
        let opt = FastOptions { settings: if has("--paper") { Settings::Paper } else { Settings::Code }, preprocess: true, ..Default::default() };
        let f = selection_qr_fast(&d, &y, &x, &b, &opt).unwrap_or_else(|e| die(&e.to_string()));
        (f.rho, f.beta, f.taus, f.gamma, None)
    } else if let Some(reps) = get("--bootstrap") {
        let reps: usize = reps.parse().unwrap_or_else(|_| die("--bootstrap needs a number"));
        let seed: u64 = get("--seed").map_or(2017, |s| s.parse().unwrap_or_else(|_| die("--seed needs a number")));
        let bs = selection_qr_bootstrap(&d, &y, &x, &b, reps, seed, &SelectionOptions::default()).unwrap_or_else(|e| die(&e.to_string()));
        eprintln!("rho standard error (weighted bootstrap, {reps} draws): {:.4}", bs.rho_se);
        let e = bs.estimate;
        (e.rho, e.beta, e.taus, e.gamma, Some(bs.beta_se))
    } else {
        let f = selection_qr(&d, &y, &x, &b, &SelectionOptions::default()).unwrap_or_else(|e| die(&e.to_string()));
        (f.rho, f.beta, f.taus, f.gamma, None)
    };
    eprintln!("rho = {rho}");
    eprintln!("probit (constant first): {gamma:?}");

    let names: Vec<String> = std::iter::once("constant".to_string()).chain(xs.split(',').map(String::from)).collect();
    let mut out: Box<dyn Write> = match get("--out") {
        Some(p) => Box::new(std::fs::File::create(&p).unwrap_or_else(|e| die(&format!("cannot write {p}: {e}")))),
        None => Box::new(std::io::stdout()),
    };
    let mut head = vec!["tau".to_string()];
    head.extend(names.iter().cloned());
    if se.is_some() {
        head.extend(names.iter().map(|n| format!("se_{n}")));
    }
    writeln!(out, "{}", head.join(",")).unwrap();
    for (j, tau) in taus.iter().enumerate() {
        let mut row = vec![format!("{tau}")];
        row.extend((0..beta.nrow).map(|k| format!("{}", beta.get(k, j))));
        if let Some(s) = &se {
            row.extend((0..s.nrow).map(|k| format!("{}", s.get(k, j))));
        }
        writeln!(out, "{}", row.join(",")).unwrap();
    }
}
