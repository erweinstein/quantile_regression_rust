//! linalg.rs -- the small dense linear algebra the solvers need: a row-major matrix type, least squares by
//! Householder QR (what MATLAB's backslash does for a rectangular system, and what numpy's lstsq computes by
//! SVD), Cholesky and LU solves for p x p systems, and a rank check. p (the number of coefficients) is small
//! in quantile regression, so nothing here is blocked or tuned; the work that matters is the O(n) loops in
//! the interior point solvers.
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

/// A dense row-major matrix: element (i, j) is `data[i * ncol + j]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub nrow: usize,
    pub ncol: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        Matrix { nrow, ncol, data: vec![0.0; nrow * ncol] }
    }
    /// From row-major data; panics if the length is not nrow * ncol.
    pub fn from_row_major(nrow: usize, ncol: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), nrow * ncol, "Matrix::from_row_major: data has the wrong length");
        Matrix { nrow, ncol, data }
    }
    /// From a list of columns of equal length.
    pub fn from_columns(cols: &[&[f64]]) -> Self {
        let ncol = cols.len();
        let nrow = if ncol == 0 { 0 } else { cols[0].len() };
        let mut m = Matrix::zeros(nrow, ncol);
        for (j, c) in cols.iter().enumerate() {
            assert_eq!(c.len(), nrow, "Matrix::from_columns: columns differ in length");
            for i in 0..nrow {
                m.data[i * ncol + j] = c[i];
            }
        }
        m
    }
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.ncol + j]
    }
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.ncol..(i + 1) * self.ncol]
    }
    /// The rows listed in `idx`, in that order.
    pub fn select_rows(&self, idx: &[usize]) -> Matrix {
        let mut data = Vec::with_capacity(idx.len() * self.ncol);
        for &i in idx {
            data.extend_from_slice(self.row(i));
        }
        Matrix { nrow: idx.len(), ncol: self.ncol, data }
    }
    /// A column of ones followed by the columns of `self` (MATLAB's [ones(n,1) X]).
    pub fn with_intercept(&self) -> Matrix {
        let p = self.ncol + 1;
        let mut data = Vec::with_capacity(self.nrow * p);
        for i in 0..self.nrow {
            data.push(1.0);
            data.extend_from_slice(self.row(i));
        }
        Matrix { nrow: self.nrow, ncol: p, data }
    }
    /// The columns of `self` followed by those of `other` (numpy's hstack).
    pub fn hstack(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.nrow, other.nrow, "Matrix::hstack: row counts differ");
        let p = self.ncol + other.ncol;
        let mut data = Vec::with_capacity(self.nrow * p);
        for i in 0..self.nrow {
            data.extend_from_slice(self.row(i));
            data.extend_from_slice(other.row(i));
        }
        Matrix { nrow: self.nrow, ncol: p, data }
    }
    /// X b.
    pub fn mul_vec(&self, b: &[f64]) -> Vec<f64> {
        assert_eq!(b.len(), self.ncol);
        (0..self.nrow).map(|i| dot(self.row(i), b)).collect()
    }
    /// X' v.
    pub fn t_mul_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.nrow);
        let mut out = vec![0.0; self.ncol];
        for i in 0..self.nrow {
            let vi = v[i];
            for (o, x) in out.iter_mut().zip(self.row(i)) {
                *o += x * vi;
            }
        }
        out
    }
    /// X' diag(w) X, a symmetric p x p matrix.
    pub fn weighted_gram(&self, w: &[f64]) -> Matrix {
        let p = self.ncol;
        let mut g = Matrix::zeros(p, p);
        for i in 0..self.nrow {
            let r = self.row(i);
            let wi = w[i];
            for j in 0..p {
                let a = wi * r[j];
                for k in 0..=j {
                    g.data[j * p + k] += a * r[k];
                }
            }
        }
        for j in 0..p {
            for k in 0..j {
                g.data[k * p + j] = g.data[j * p + k];
            }
        }
        g
    }
}

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A Householder QR factorization of an n x p matrix (n >= p), stored column-major with the Householder
/// vectors below the diagonal, as LAPACK's dgeqrf does.
pub struct Qr {
    n: usize,
    p: usize,
    a: Vec<f64>, // column-major n x p
    tau: Vec<f64>,
    rdiag: Vec<f64>,
}

impl Qr {
    /// Factor the n x p matrix whose row i is `scale[i] * x.row(i)` (scale = None: x itself).
    pub fn new(x: &Matrix, scale: Option<&[f64]>) -> Qr {
        let (n, p) = (x.nrow, x.ncol);
        assert!(n >= p, "Qr::new needs at least as many rows as columns");
        let mut a = vec![0.0; n * p];
        for i in 0..n {
            let s = scale.map_or(1.0, |s| s[i]);
            let r = x.row(i);
            for j in 0..p {
                a[j * n + i] = s * r[j];
            }
        }
        let mut tau = vec![0.0; p];
        let mut rdiag = vec![0.0; p];
        for k in 0..p {
            let col = &mut a[k * n..(k + 1) * n];
            let norm = col[k..].iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm == 0.0 {
                tau[k] = 0.0;
                rdiag[k] = 0.0;
                continue;
            }
            let alpha = if col[k] > 0.0 { -norm } else { norm };
            // v = x - alpha e1, normalized so v[k] = 1
            let v0 = col[k] - alpha;
            for v in col[k + 1..].iter_mut() {
                *v /= v0;
            }
            tau[k] = -v0 / alpha;
            col[k] = alpha;
            rdiag[k] = alpha;
            // apply H = I - tau v v' to the remaining columns
            let (head, tail) = a.split_at_mut((k + 1) * n);
            let vk = &head[k * n..(k + 1) * n];
            for j in 0..(p - k - 1) {
                let cj = &mut tail[j * n..(j + 1) * n];
                let mut s = cj[k];
                for i in k + 1..n {
                    s += vk[i] * cj[i];
                }
                s *= tau[k];
                cj[k] -= s;
                for i in k + 1..n {
                    cj[i] -= s * vk[i];
                }
            }
        }
        Qr { n, p, a, tau, rdiag }
    }

    /// The least-squares solution of A b = rhs (A the factored matrix).
    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let (n, p) = (self.n, self.p);
        assert_eq!(rhs.len(), n);
        let mut qtb = rhs.to_vec();
        for k in 0..p {
            if self.tau[k] == 0.0 {
                continue;
            }
            let v = &self.a[k * n..(k + 1) * n];
            let mut s = qtb[k];
            for i in k + 1..n {
                s += v[i] * qtb[i];
            }
            s *= self.tau[k];
            qtb[k] -= s;
            for i in k + 1..n {
                qtb[i] -= s * v[i];
            }
        }
        let mut b = vec![0.0; p];
        for k in (0..p).rev() {
            let mut s = qtb[k];
            for j in k + 1..p {
                s -= self.a[j * n + k] * b[j];
            }
            b[k] = s / self.rdiag[k];
        }
        b
    }

    /// Numerical rank: the number of |R_kk| above max|R_kk| * max(n, p) * eps (numpy's matrix_rank uses the
    /// same tolerance on singular values).
    pub fn rank(&self) -> usize {
        let mx = self.rdiag.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let tol = mx * (self.n.max(self.p) as f64) * f64::EPSILON;
        self.rdiag.iter().filter(|v| v.abs() > tol).count()
    }
}

/// Least squares: argmin_b || diag(scale) (X b - rhs/scale) ||, i.e. solve (diag(scale) X) b = rhs.
pub fn lstsq(x: &Matrix, scale: Option<&[f64]>, rhs: &[f64]) -> Vec<f64> {
    Qr::new(x, scale).solve(rhs)
}

/// Solve the symmetric positive definite system M b = rhs by Cholesky; None if M is not positive definite.
pub fn cholesky_solve(m: &Matrix, rhs: &[f64]) -> Option<Vec<f64>> {
    let p = m.nrow;
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut d = m.get(j, j);
        for k in 0..j {
            d -= l[j * p + k] * l[j * p + k];
        }
        if !(d > 0.0) || !d.is_finite() {
            return None;
        }
        let d = d.sqrt();
        l[j * p + j] = d;
        for i in j + 1..p {
            let mut s = m.get(i, j);
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            l[i * p + j] = s / d;
        }
    }
    let mut z = rhs.to_vec();
    for i in 0..p {
        for k in 0..i {
            z[i] -= l[i * p + k] * z[k];
        }
        z[i] /= l[i * p + i];
    }
    for i in (0..p).rev() {
        for k in i + 1..p {
            z[i] -= l[k * p + i] * z[k];
        }
        z[i] /= l[i * p + i];
    }
    Some(z)
}

/// Solve the square system M b = rhs by Gaussian elimination with partial pivoting (as LAPACK's dgesv,
/// which numpy's solve calls). None if M is singular.
pub fn lu_solve(m: &Matrix, rhs: &[f64]) -> Option<Vec<f64>> {
    let p = m.nrow;
    assert_eq!(m.ncol, p);
    let mut a = m.data.clone();
    let mut b = rhs.to_vec();
    for k in 0..p {
        let mut piv = k;
        for i in k + 1..p {
            if a[i * p + k].abs() > a[piv * p + k].abs() {
                piv = i;
            }
        }
        if a[piv * p + k] == 0.0 {
            return None;
        }
        if piv != k {
            for j in 0..p {
                a.swap(k * p + j, piv * p + j);
            }
            b.swap(k, piv);
        }
        for i in k + 1..p {
            let f = a[i * p + k] / a[k * p + k];
            a[i * p + k] = f;
            for j in k + 1..p {
                a[i * p + j] -= f * a[k * p + j];
            }
            b[i] -= f * b[k];
        }
    }
    for i in (0..p).rev() {
        let mut s = b[i];
        for j in i + 1..p {
            s -= a[i * p + j] * b[j];
        }
        b[i] = s / a[i * p + i];
    }
    Some(b)
}

/// The inverse of a square matrix (by LU, column by column); None if singular.
pub fn inverse(m: &Matrix) -> Option<Matrix> {
    let p = m.nrow;
    let mut inv = Matrix::zeros(p, p);
    for j in 0..p {
        let mut e = vec![0.0; p];
        e[j] = 1.0;
        let c = lu_solve(m, &e)?;
        for i in 0..p {
            inv.data[i * p + j] = c[i];
        }
    }
    Some(inv)
}

/// A B for square or rectangular matrices.
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.ncol, b.nrow);
    let mut c = Matrix::zeros(a.nrow, b.ncol);
    for i in 0..a.nrow {
        for k in 0..a.ncol {
            let aik = a.get(i, k);
            for j in 0..b.ncol {
                c.data[i * b.ncol + j] += aik * b.get(k, j);
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_least_squares_matches_normal_equations() {
        let x = Matrix::from_row_major(5, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
        let y = [1.0, 2.9, 5.1, 7.0, 9.2];
        let b = lstsq(&x, None, &y);
        let g = x.weighted_gram(&[1.0; 5]);
        let b2 = cholesky_solve(&g, &x.t_mul_vec(&y)).unwrap();
        for (u, v) in b.iter().zip(&b2) {
            assert!((u - v).abs() < 1e-12);
        }
        assert_eq!(Qr::new(&x, None).rank(), 2);
    }

    #[test]
    fn lu_and_inverse() {
        let m = Matrix::from_row_major(3, 3, vec![0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 3.0, 0.0, 1.0]);
        let b = lu_solve(&m, &[3.0, 2.0, 4.0]).unwrap();
        let r = m.mul_vec(&b);
        for (u, v) in r.iter().zip(&[3.0, 2.0, 4.0]) {
            assert!((u - v).abs() < 1e-12);
        }
        let inv = inverse(&m).unwrap();
        let id = matmul(&m, &inv);
        for i in 0..3 {
            for j in 0..3 {
                assert!((id.get(i, j) - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }
}
