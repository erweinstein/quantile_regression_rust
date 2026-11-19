//! Stage 4: the extensions (qr_extensions.py, selection_qr_fast.py, smoothed_qr.py) against the Python port.
mod common;
use common::*;
use quantile_regression::extensions::{rearrange, rq_se, SeMethod};
use quantile_regression::fast::{rq_preprocessed_default, selection_qr_fast, FastOptions, Search, Settings};
use quantile_regression::rq::{rq_objective, Tau};
use quantile_regression::smoothed::conquer;

#[test]
fn rq_se_matches_python() {
    let r = load("extensions.json");
    let s = &r["se"];
    let (x, y) = (mat(&s["X"]), vec(&s["y"]));
    for c in s["cases"].as_array().unwrap() {
        let tau = c["tau"].as_f64().unwrap();
        for (m, key) in [(SeMethod::Ker, "ker"), (SeMethod::Iid, "iid")] {
            let (se, _) = rq_se(&x, &y, tau, None, m, 0.05).unwrap();
            let py = vec(&c[key]);
            let rel = se.iter().zip(&py).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
            println!("rq_se {key} tau {tau}: rel {rel:.1e}");
            assert!(rel < 1e-9, "{se:?} {py:?}");
        }
    }
}

#[test]
fn preprocessing_matches_python() {
    let r = load("extensions.json");
    let p = &r["preprocessed"];
    let (x, y, tau, b0) = (mat(&p["X"]), vec(&p["y"]), vec(&p["tau"]), vec(&p["b0"]));
    let (b, info) = rq_preprocessed_default(&x, &y, Tau::PerObs(&tau), &b0);
    let obj = rq_objective(&x, &y, &b, &tau);
    let (obj_py, obj_full) = (p["objective"].as_f64().unwrap(), p["objective_full"].as_f64().unwrap());
    println!("preprocessed: {info:?}; coef diff {:.1e}; objective {obj} (Python {obj_py}, full problem {obj_full})",
             max_abs_diff(&b, &vec(&p["coef"])));
    assert!(!info.fallback);
    assert!(max_abs_diff(&b, &vec(&p["coef"])) < 1e-6);
    assert!((obj - obj_full).abs() / obj_full < 1e-8);
}

#[test]
fn selection_qr_fast_variants_match_python() {
    let r = load("extensions.json");
    let f = &r["fast"];
    let (d, y, x, b) = (vec(&f["D"]), vec(&f["Y"]), col(&f["X"]), col(&f["B"]));
    let variants = [
        ("exact", FastOptions::default()),
        ("pre", FastOptions { preprocess: true, ..Default::default() }),
        ("pre_coarse", FastOptions { preprocess: true, search: Search::coarse(), ..Default::default() }),
        ("paper_pre_coarse", FastOptions { settings: Settings::Paper, preprocess: true, search: Search::coarse(), ..Default::default() }),
        ("tie", FastOptions { tie_tol: 1e-7, ..Default::default() }),
    ];
    for (name, opt) in variants {
        let fit = selection_qr_fast(&d, &y, &x, &b, &opt).unwrap();
        let py = &f[name];
        let rho_py = py["rho"].as_f64().unwrap();
        let db = max_abs_diff(&fit.beta.data, &mat(&py["beta"]).data);
        // evaluated grid points: same set, and close values
        let obj_py = vec(&py["objective"]);
        let same_set = fit.objective.iter().zip(&obj_py).all(|(a, b)| a.is_nan() == (*b == -1.0));
        println!("{name}: rho {} (Python {rho_py}); beta diff {db:.1e}; {} grid points evaluated; same set {same_set}",
                 fit.rho, fit.evaluated);
        assert!(same_set);
        assert_eq!(fit.rho, rho_py, "{name}");
        assert!(db < 1e-6, "{name}: {db:e}");
    }
}

#[test]
fn conquer_matches_python() {
    let r = load("extensions.json");
    let c = &r["conquer"];
    let (x, y) = (mat(&c["X"]), vec(&c["y"]));
    for case in c["cases"].as_array().unwrap() {
        let tau = case["tau"].as_f64().unwrap();
        let b = conquer(&x, &y, tau, None, None, None, 1e-8, 100);
        let d = max_abs_diff(&b, &vec(&case["coef"]));
        println!("conquer tau {tau}: {d:.1e}");
        assert!(d < 1e-6); // Newton stops when the step is below 1e-8, so the two agree to about that
    }
}

#[test]
fn rearrange_matches_python() {
    let r = load("extensions.json");
    let a = &r["rearrange"];
    let q = rearrange(&mat(&a["beta"]), &mat(&a["x"]));
    assert!(max_abs_diff(&q.data, &mat(&a["curves"]).data) < 1e-15);
}
