//! normal.rs -- the standard normal cdf, density and quantile function, and the standard bivariate normal cdf.
//!
//! Phi uses libm's erfc (a port of musl's, accurate to about one unit in the last place); Phi^{-1} refines a
//! starting value by Halley's method on that Phi, in the lower tail so that small probabilities keep full
//! relative accuracy. (statrs's erfc was tried first and is accurate only to about 1e-10.) The bivariate normal cdf replaces MATLAB's
//! copulacdf('Gaussian', ...) in Code_QRS.m; the Python port computes it with Owen's T function (scipy), and
//! here it is computed by Genz's (2004) algorithm (his Fortran BVND, as revised from Drezner and Wesolowsky
//! 1990), with Gauss-Legendre rules of 6, 12 or 20 points depending on |rho|. The two agree to about 1e-15
//! (tests/stage2_selection.rs).
//!
//! References
//! - Drezner Z, Wesolowsky GO (1990). On the computation of the bivariate normal integral. Journal of
//!   Statistical Computation and Simulation 35(1-2), 101-107.
//! - Genz A (2004). Numerical computation of rectangular bivariate and trivariate normal and t probabilities.
//!   Statistics and Computing 14(3), 251-260.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use libm::erfc;
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::sync::OnceLock;

/// Standard normal cdf Phi(x).
#[inline]
pub fn cdf(x: f64) -> f64 {
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// Standard normal density phi(x).
#[inline]
pub fn pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal quantile function Phi^{-1}(u), for 0 <= u <= 1 (-inf and +inf at the ends).
pub fn ppf(u: f64) -> f64 {
    if u.is_nan() {
        return f64::NAN;
    }
    if u <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if u >= 1.0 {
        return f64::INFINITY;
    }
    if u > 0.5 {
        return -ppf(1.0 - u); // 1 - u is exact for u >= 0.5
    }
    if u == 0.5 {
        return 0.0;
    }
    // starting value: the tail approximation x = -sqrt(-2 log u) corrected (u < 0.5 here)
    let t = (-2.0 * u.ln()).sqrt();
    let mut x = -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t));
    // Halley's method on Phi(x) - u, with the relative residual
    for _ in 0..50 {
        let f = cdf(x) - u;
        let r = f / pdf(x);
        let dx = r / (1.0 + 0.5 * x * r);
        x -= dx;
        if dx.abs() <= 1e-16 * x.abs().max(1.0) {
            break;
        }
    }
    x
}

/// Half of a Gauss-Legendre rule on [-1, 1]: the positive nodes and their weights.
fn gauss_legendre_half(m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    for i in 1..=m / 2 {
        // Newton's method on the Legendre polynomial P_m from the usual starting value
        let mut x = (PI * (i as f64 - 0.25) / (m as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=m {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = m as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-16 {
                break;
            }
        }
        let (mut p0, mut p1) = (1.0, x);
        for k in 2..=m {
            let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
            p0 = p1;
            p1 = p2;
        }
        let dp = m as f64 * (x * p1 - p0) / (x * x - 1.0);
        nodes.push(x);
        weights.push(2.0 / ((1.0 - x * x) * dp * dp));
    }
    (nodes, weights)
}

fn rules() -> &'static [(Vec<f64>, Vec<f64>); 3] {
    static R: OnceLock<[(Vec<f64>, Vec<f64>); 3]> = OnceLock::new();
    R.get_or_init(|| [gauss_legendre_half(6), gauss_legendre_half(12), gauss_legendre_half(20)])
}

/// Genz's BVNU for one correlation r, with everything that depends only on r computed once (the sines of the
/// quadrature nodes for |r| < 0.925, and the nodes' xs and sqrt(1 - xs) otherwise). `upper` then gives the same
/// numbers as evaluating the whole algorithm for each (h, k), with far fewer sin and sqrt calls; the rotated
/// quantile regressions evaluate it once per observation at each of hundreds of (tau, rho) pairs.
pub struct Bvn {
    r: f64,
    /// |r| < 0.925: (weight, sin at node 1 - x, sin at node 1 + x), and asin(r)
    small: Vec<(f64, f64, f64)>,
    asr: f64,
    /// |r| >= 0.925: (weight, xs, rs) for each node and sign, and a = sqrt(1 - r^2) / 2
    large: Vec<(f64, f64, f64)>,
}

impl Bvn {
    pub fn new(r: f64) -> Bvn {
        let rr = rules();
        let i = if r.abs() < 0.3 {
            0
        } else if r.abs() < 0.75 {
            1
        } else {
            2
        };
        let (x, w) = (&rr[i].0, &rr[i].1);
        let mut b = Bvn { r, small: vec![], asr: 0.0, large: vec![] };
        if r.abs() < 0.925 {
            b.asr = r.asin();
            for (xi, wi) in x.iter().zip(w) {
                b.small.push((*wi, (b.asr * (1.0 - xi) / 2.0).sin(), (b.asr * (1.0 + xi) / 2.0).sin()));
            }
        } else if r.abs() < 1.0 {
            let a = ((1.0 - r) * (1.0 + r)).sqrt() / 2.0;
            for (xi, wi) in x.iter().zip(w) {
                for sgn in [-1.0, 1.0] {
                    let xs = (a + a * sgn * xi).powi(2);
                    b.large.push((*wi, xs, (1.0 - xs).sqrt()));
                }
            }
        }
        b
    }

    /// P(X > h, Y > k).
    pub fn upper(&self, h: f64, k: f64) -> f64 {
        let r = self.r;
        if h == f64::INFINITY || k == f64::INFINITY {
            return 0.0;
        }
        if h == f64::NEG_INFINITY {
            return if k == f64::NEG_INFINITY { 1.0 } else { cdf(-k) };
        }
        if k == f64::NEG_INFINITY {
            return cdf(-h);
        }
        let mut k = k;
        let mut hk = h * k;
        let mut bvn = 0.0;
        if r.abs() < 0.925 {
            let hs = (h * h + k * k) / 2.0;
            for &(wi, sn1, sn2) in &self.small {
                bvn += wi * ((sn1 * hk - hs) / (1.0 - sn1 * sn1)).exp();
                bvn += wi * ((sn2 * hk - hs) / (1.0 - sn2 * sn2)).exp();
            }
            bvn = bvn * self.asr / (4.0 * PI) + cdf(-h) * cdf(-k);
        } else {
            if r < 0.0 {
                k = -k;
                hk = -hk;
            }
            if r.abs() < 1.0 {
                let as_ = (1.0 - r) * (1.0 + r);
                let mut a = as_.sqrt();
                let bs = (h - k) * (h - k);
                let c = (4.0 - hk) / 8.0;
                let d = (12.0 - hk) / 16.0;
                let asr = -(bs / as_ + hk) / 2.0;
                if asr > -100.0 {
                    bvn = a * asr.exp() * (1.0 - c * (bs - as_) * (1.0 - d * bs / 5.0) / 3.0 + c * d * as_ * as_ / 5.0);
                }
                if hk > -100.0 {
                    let b = bs.sqrt();
                    let sp = (2.0 * PI).sqrt() * cdf(-b / a);
                    bvn -= (-hk / 2.0).exp() * sp * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
                }
                a /= 2.0;
                for &(wi, xs, rs) in &self.large {
                    let asr = -(bs / xs + hk) / 2.0;
                    if asr > -100.0 {
                        let sp = 1.0 + c * xs * (1.0 + d * xs);
                        let ep = (-hk * xs / (2.0 * (1.0 + rs) * (1.0 + rs))).exp() / rs;
                        bvn += a * wi * asr.exp() * (ep - sp);
                    }
                }
                bvn = -bvn / (2.0 * PI);
            }
            if r > 0.0 {
                bvn += cdf(-h.max(k));
            } else if h >= k {
                bvn = -bvn;
            } else {
                let l = if h < 0.0 { cdf(k) - cdf(h) } else { cdf(-h) - cdf(-k) };
                bvn = l - bvn;
            }
        }
        bvn.clamp(0.0, 1.0)
    }

    /// Phi_2(h, k; r) = P(X <= h, Y <= k).
    #[inline]
    pub fn cdf(&self, h: f64, k: f64) -> f64 {
        self.upper(-h, -k)
    }
}

/// P(X > h, Y > k) for a standard bivariate normal pair with correlation r (Genz's BVNU).
pub fn bvn_upper(h: f64, k: f64, r: f64) -> f64 {
    Bvn::new(r).upper(h, k)
}

/// The standard bivariate normal cdf Phi_2(h, k; rho) = P(X <= h, Y <= k).
#[inline]
pub fn bvn_cdf(h: f64, k: f64, rho: f64) -> f64 {
    bvn_upper(-h, -k, rho)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_six_points() {
        // Abramowitz and Stegun, Table 25.4
        let (x, w) = gauss_legendre_half(6);
        let xr = [0.932469514203152, 0.661209386466265, 0.238619186083197];
        let wr = [0.171324492379170, 0.360761573048139, 0.467913934572691];
        for i in 0..3 {
            assert!((x[i] - xr[i]).abs() < 1e-14 && (w[i] - wr[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn bvn_special_values() {
        // Phi_2(0, 0; rho) = 1/4 + asin(rho) / (2 pi)
        for rho in [-0.99, -0.5, 0.0, 0.3, 0.95] {
            assert!((bvn_cdf(0.0, 0.0, rho) - (0.25 + rho.asin() / (2.0 * PI))).abs() < 1e-15);
        }
        // independence
        assert!((bvn_cdf(0.3, -1.2, 0.0) - cdf(0.3) * cdf(-1.2)).abs() < 1e-15);
    }
}
