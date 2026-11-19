//! Stage 2: normal functions, the bivariate normal cdf, the probit and the selection-corrected estimator
//! (selection_qr.py, qr_extensions.py) against the Python port.
mod common;
use common::*;
use quantile_regression::normal;
use quantile_regression::selection::{default_taus, default_vecrho, probit, selection_qr, selection_qr_weighted, weighted_rq, SelectionOptions};

#[test]
fn normal_functions_match_scipy() {
    let r = load("normal.json");
    let (x, u) = (vec(&r["x"]), vec(&r["u"]));
    let dc = x.iter().zip(vec(&r["cdf"])).map(|(x, c)| ((normal::cdf(*x) - c) / c.max(1e-300)).abs()).fold(0.0, f64::max);
    let dp = max_abs_diff(&x.iter().map(|v| normal::pdf(*v)).collect::<Vec<_>>(), &vec(&r["pdf"]));
    let dq = max_abs_diff(&u.iter().map(|v| normal::ppf(*v)).collect::<Vec<_>>(), &vec(&r["ppf"]));
    println!("Phi rel {dc:.2e}, phi {dp:.2e}, Phi^-1 {dq:.2e}");
    assert!(dc < 1e-13 && dp < 1e-15 && dq < 1e-13);
}

#[test]
fn bivariate_normal_matches_owens_t_version() {
    let r = load("normal.json");
    let (h, k) = (vec(&r["h"]), vec(&r["k"]));
    let mut worst = 0.0f64;
    for case in r["bvn"].as_array().unwrap() {
        let rho = case["rho"].as_f64().unwrap();
        let ours: Vec<f64> = h.iter().zip(&k).map(|(a, b)| normal::bvn_cdf(*a, *b, rho)).collect();
        let d = max_abs_diff(&ours, &vec(&case["values"]));
        println!("rho {rho:+.3}: {d:.2e}");
        worst = worst.max(d);
    }
    assert!(worst < 1e-12, "{worst:e}");
}

#[test]
fn grids_are_code_qrs_grids() {
    let r = load("selection.json");
    let c = &r["cases"][0];
    assert_eq!(default_vecrho(), vec(&c["vecrho"]));
    assert_eq!(default_taus(), vec(&c["taus"]));
}

#[test]
fn selection_qr_matches_python() {
    let r = load("selection.json");
    for c in r["cases"].as_array().unwrap() {
        let (d, y, x, b) = (vec(&c["D"]), vec(&c["Y"]), col(&c["X"]), col(&c["B"]));
        let z = x.hstack(&b);
        let g = probit(&d, &z, None).unwrap();
        let dg = max_abs_diff(&g, &vec(&c["gamma"]));
        let f = selection_qr(&d, &y, &x, &b, &SelectionOptions::default()).unwrap();
        let rho_py = c["rho"].as_f64().unwrap();
        let obj_py = vec(&c["objective"]);
        let dobj = f.objective.iter().zip(&obj_py).map(|(a, b)| (a - b).abs() / b.abs().max(1e-12)).fold(0.0, f64::max);
        let beta_py = mat(&c["beta"]);
        let db = max_abs_diff(&f.beta.data, &beta_py.data);
        println!("n {} seed {}: gamma {dg:.1e}; rho {} (Python {rho_py}); objective rel {dobj:.1e}; beta {db:.1e}",
                 c["n"], c["seed"], f.rho);
        assert!(dg < 1e-12);
        assert_eq!(f.rho, rho_py);
        assert!(db < 1e-8);
    }
}

#[test]
fn weighted_estimator_matches_python() {
    let r = load("weighted.json");
    let (d, y, x, b, w) = (vec(&r["D"]), vec(&r["Y"]), col(&r["X"]), col(&r["B"]), vec(&r["w"]));
    let f = selection_qr_weighted(&d, &y, &x, &b, Some(&w), &SelectionOptions::default()).unwrap();
    let db = max_abs_diff(&f.beta.data, &mat(&r["beta"]).data);
    println!("weighted: gamma {:.1e}; rho {} (Python {}); beta {db:.1e}", max_abs_diff(&f.gamma, &vec(&r["gamma"])), f.rho, r["rho"]);
    assert_eq!(f.rho, r["rho"].as_f64().unwrap());
    assert!(db < 1e-8);
    let q = &r["rq"];
    let bw = weighted_rq(&mat(&q["X"]), &vec(&q["y"]), q["tau"].as_f64().unwrap(), &vec(&q["w"])).unwrap();
    assert!(max_abs_diff(&bw, &vec(&q["coef"])) < 1e-8);
}

#[test]
fn bootstrap_does_not_depend_on_the_number_of_threads() {
    use quantile_regression::selection::selection_qr_bootstrap;
    let r = load("weighted.json");
    let (d, y, x, b) = (vec(&r["D"]), vec(&r["Y"]), col(&r["X"]), col(&r["B"]));
    let opt = SelectionOptions { vecrho: (0..20).map(|i| -0.95 + 0.1 * i as f64).collect(), ..Default::default() };
    let run = |threads| {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()
            .install(|| selection_qr_bootstrap(&d, &y, &x, &b, 6, 42, &opt).unwrap())
    };
    let (a, c) = (run(1), run(2));
    assert_eq!(a.rho_draws, c.rho_draws);
    assert_eq!(a.beta_se.data, c.beta_se.data);
    assert!(a.rho_se > 0.0 && a.beta_se.data.iter().all(|v| *v > 0.0));
}
