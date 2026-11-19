//! Stage 1: rq, rq_fast, rqic and the objective against the Python port (rq.py, selection_qr_fast.py).
mod common;
use common::*;
use quantile_regression::rq::{rq, rq_fast, rq_objective, rqic};

#[test]
fn rq_matches_python() {
    let r = load("rq.json");
    let mut worst = (0.0f64, 0.0f64, 0.0f64);
    for case in r["cases"].as_array().unwrap() {
        let (x, y) = (mat(&case["X"]), vec(&case["y"]));
        for f in case["fits"].as_array().unwrap() {
            let tau = f["tau"].as_f64().unwrap();
            let b = rq(&x, &y, tau).unwrap();
            let bf = rq_fast(&x, &y, tau).unwrap();
            let obj = rq_objective(&x, &y, &b, tau);
            worst.0 = worst.0.max(max_abs_diff(&b, &vec(&f["coef"])));
            worst.1 = worst.1.max(max_abs_diff(&bf, &vec(&f["coef_fast"])));
            worst.2 = worst.2.max((obj - f["objective"].as_f64().unwrap()).abs() / obj.abs());
        }
    }
    println!("rq: max |coef diff| {:.2e}, rq_fast {:.2e}, rel. objective {:.2e}", worst.0, worst.1, worst.2);
    assert!(worst.0 < 1e-8 && worst.1 < 1e-8 && worst.2 < 1e-10, "{worst:?}");
}

#[test]
fn rq_observation_specific_tau_matches_python() {
    let r = load("rq.json");
    let o = &r["per_obs"];
    let (x, y, tau) = (mat(&o["X"]), vec(&o["y"]), vec(&o["tau"]));
    let b = rq(&x, &y, &tau).unwrap();
    let bf = rq_fast(&x, &y, &tau).unwrap();
    let d = max_abs_diff(&b, &vec(&o["coef"]));
    let df = max_abs_diff(&bf, &vec(&o["coef_fast"]));
    println!("per-observation tau: {d:.2e}, fast {df:.2e}");
    assert!(d < 1e-8 && df < 1e-8);
}

#[test]
fn rqic_matches_python() {
    let r = load("rq.json");
    let c = &r["rqic"];
    let (x, y) = (mat(&c["X"]), vec(&c["y"]));
    for case in c["cases"].as_array().unwrap() {
        let b = rqic(&x, &y, &mat(&case["R"]), &vec(&case["r"]), 0.5).unwrap();
        let d = max_abs_diff(&b, &vec(&case["coef"]));
        println!("rqic {:?}: {d:.2e}", vec(&case["r"]));
        assert!(d < 1e-8, "{b:?}");
    }
}
