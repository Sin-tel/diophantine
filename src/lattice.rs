//! Lattice basis reduction, CVP, SVP.
//!
//! Uses `f64` for calculations, so the results are not exact.

use crate::{DiophantineError, Matrix};
use std::ops::SubAssign;

/// Subtracts `scale * other` from `target` in place.
fn vec_sub_assign<T>(target: &mut [T], other: &[T], scale: T)
where
    T: Copy + SubAssign + std::ops::Mul<Output = T>,
{
    for (t, &o) in target.iter_mut().zip(other.iter()) {
        *t -= scale * o;
    }
}

fn inner_prod(x: &[f64], y: &[f64], w: &Matrix<f64>) -> f64 {
    let n = w.len();
    if n == 0 {
        return 0.0;
    }
    let m = w[0].len();
    assert_eq!(x.len(), n);
    assert_eq!(y.len(), m);

    // temp = W * y
    let mut temp = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..m {
            sum += w[i][j] * y[j];
        }
        temp[i] = sum;
    }

    // result = x dot temp
    let mut res = 0.0;
    for i in 0..n {
        res += x[i] * temp[i];
    }
    res
}

/// Gram-Schmidt Orthogonalization
fn gramschmidt(v: &Matrix<i64>, w: &Matrix<f64>) -> Matrix<f64> {
    let nrows = v.len();
    if nrows == 0 {
        return vec![];
    }

    // Convert input integer basis to f64
    let v_f64: Matrix<f64> = v
        .iter()
        .map(|row| row.iter().map(|&x| x as f64).collect())
        .collect();

    let mut u = v_f64.clone();

    for i in 1..nrows {
        // Clone current row to work on it, then assign back
        let mut ui = u[i].clone();
        let v_i = &v_f64[i];

        for j in 0..i {
            let uj = &u[j];

            let num = inner_prod(uj, v_i, w);
            let den = inner_prod(uj, uj, w);

            // Handle zero-norm vectors if necessary (though rare in basis)
            let proj_coeff = if den.abs() < 1e-9 { 0.0 } else { num / den };

            vec_sub_assign(&mut ui, uj, proj_coeff);
        }
        u[i] = ui;
    }
    u
}

// Helper for LLL: Calculate mu coefficient
fn mu(basis: &Matrix<i64>, ortho: &Matrix<f64>, w: &Matrix<f64>, i: usize, j: usize) -> f64 {
    let a = &ortho[j];
    // Convert basis row to f64 on the fly for calculation
    let b: Vec<f64> = basis[i].iter().map(|&x| x as f64).collect();

    let num = inner_prod(a, &b, w);
    let den = inner_prod(a, a, w);

    if den.abs() < 1e-9 { 0.0 } else { num / den }
}

/// Compute the LLL reduction of a basis.
///
/// Returns an error if dimensions do not match.
///
/// # Arguments
/// * `basis` - The lattice basis (row vectors).
/// * `delta` - The reduction parameter (typically 0.75 or 0.99).
/// * `w` - The quadratic form matrix (weights). Pass Identity matrix for standard Euclidean.
pub fn lll(
    basis: &Matrix<i64>,
    delta: f64,
    w: &Matrix<f64>,
) -> Result<Matrix<i64>, DiophantineError> {
    let mut basis = basis.clone();

    let n = basis.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let m = basis[0].len();

    if w.len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "Basis should have same number of columns as w".to_string(),
        ));
    }

    if w.len() != w[0].len() {
        return Err(DiophantineError::InvalidDimensions(
            "W must be square".to_string(),
        ));
    }

    let mut ortho = gramschmidt(&basis, w);
    let mut k = 1;

    while k < n {
        // Size reduction step
        for j in (0..k).rev() {
            let mu_kj = mu(&basis, &ortho, w, k, j);
            if mu_kj.abs() > 0.5 {
                let mu_int = mu_kj.round_ties_even() as i64;

                // basis[k] -= mu_int * basis[j]
                let basis_j = basis[j].clone();
                vec_sub_assign(&mut basis[k], &basis_j, mu_int);

                // Update GS
                ortho = gramschmidt(&basis, w);
            }
        }

        // LLL condition check
        let mu_k_k1 = mu(&basis, &ortho, w, k, k - 1);
        let norm_ortho_k1 = inner_prod(&ortho[k - 1], &ortho[k - 1], w);
        let norm_ortho_k = inner_prod(&ortho[k], &ortho[k], w);

        let l_condition = (delta - mu_k_k1.powi(2)) * norm_ortho_k1;

        if norm_ortho_k >= l_condition {
            k += 1;
        } else {
            // Swap rows k and k-1
            basis.swap(k, k - 1);

            // Recompute GS
            ortho = gramschmidt(&basis, w);

            k = k.saturating_sub(1).max(1);
        }
    }

    Ok(basis)
}

/// Babai's Nearest Plane Algorithm for approximate CVP.
///
/// Returns an error if dimensions do not match.
/// The basis should be LLL-reduced first.
///
/// # Arguments
/// * `v`     - The query vector.
/// * `basis` - The lattice basis (row vectors).
/// * `w` - The quadratic form matrix (weights). Pass Identity matrix for standard Euclidean.
pub fn nearest_plane(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
) -> Result<Vec<i64>, DiophantineError> {
    let mut b = v.to_vec();

    let n = basis.len(); // number of rows
    if n == 0 {
        return Ok(vec![]);
    }
    let m = basis[0].len(); // number of cols

    if v.len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "Target vector should have same length as basis".to_string(),
        ));
    }
    if w.len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "Basis should have same number of columns as w".to_string(),
        ));
    }
    if w.len() != w[0].len() {
        return Err(DiophantineError::InvalidDimensions(
            "W must be square".to_string(),
        ));
    }

    let ortho = gramschmidt(basis, w);

    for j in (0..n).rev() {
        let a = &ortho[j];
        let b_f64: Vec<f64> = b.iter().map(|&x| x as f64).collect();

        let num = inner_prod(a, &b_f64, w);
        let den = inner_prod(a, a, w);
        let mu = if den.abs() < 1e-9 { 0.0 } else { num / den };

        let mu_int = mu.round_ties_even() as i64;

        // b -= mu_int * basis[j]
        let basis_j = &basis[j];
        vec_sub_assign(&mut b, basis_j, mu_int);
    }

    let mut result = Vec::with_capacity(v.len());
    for (orig_val, residue_val) in v.iter().zip(b.iter()) {
        result.push(orig_val - residue_val);
    }
    Ok(result)
}

/// Slack on an enumeration radius, so that points exactly on the bound survive rounding.
fn with_tolerance(radius_sq: f64) -> f64 {
    radius_sq * (1.0 + 1e-9) + 1e-9
}

/// Compute d^T W d, without allocating.
fn quad_form(d: &[f64], w: &Matrix<f64>) -> f64 {
    let mut res = 0.0;
    for i in 0..d.len() {
        let mut row = 0.0;
        for j in 0..d.len() {
            row += w[i][j] * d[j];
        }
        res += d[i] * row;
    }
    res
}

/// Checks that `v`, the rows of `basis` and `w` agree on the ambient dimension.
fn check_dims(v: &[i64], basis: &Matrix<i64>, w: &Matrix<f64>) -> Result<(), DiophantineError> {
    let m = v.len();
    if basis.iter().any(|row| row.len() != m) {
        return Err(DiophantineError::InvalidDimensions(
            "Target vector should have same length as basis columns".to_string(),
        ));
    }
    if w.len() != m || w.iter().any(|row| row.len() != m) {
        return Err(DiophantineError::InvalidDimensions(
            "W must be square and match basis columns".to_string(),
        ));
    }
    Ok(())
}

/// Schnorr-Euchner enumeration.
///
/// Visits the vectors `x` of the lattice spanned by `basis` with `|v - x|^2_w <= radius_sq`
/// (up to a small tolerance), in Schnorr-Euchner order: near first, but not sorted.
/// `visit` gets the coefficients of `x`, `x` itself and `|v - x|^2_w` as computed during
/// enumeration, and returns the squared radius to continue with: the same to go on,
/// smaller to tighten, negative to stop.
///
/// The rows of `basis` must be linearly independent, and `w` must be definite on their
/// span. The basis should be reduced for speed.
pub(crate) fn enumerate<F>(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
    radius_sq: f64,
    mut visit: F,
) -> Result<(), DiophantineError>
where
    F: FnMut(&[i64], &[i64], f64) -> Result<f64, DiophantineError>,
{
    check_dims(v, basis, w)?;
    if radius_sq.is_nan() || radius_sq < 0.0 {
        return Ok(());
    }

    let n = basis.len();
    let m = v.len();
    let v_f64: Vec<f64> = v.iter().map(|&x| x as f64).collect();

    if n == 0 {
        // The lattice is just the origin
        let dist = quad_form(&v_f64, w);
        if dist <= with_tolerance(radius_sq) {
            visit(&[], &vec![0; m], dist)?;
        }
        return Ok(());
    }

    // Compute Gram-Schmidt orthogonalization
    let ortho = gramschmidt(basis, w);

    let mut b_star_norms = vec![0.0; n];
    for i in 0..n {
        let norm = inner_prod(&ortho[i], &ortho[i], w);
        // Floor the norm at a small epsilon to avoid infinite loops on degenerate dimensions
        b_star_norms[i] = if norm < 1e-9 { 1e-9 } else { norm };
    }

    // Cache floating point representation of the basis to avoid inner loop allocations
    let mut basis_f64 = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            basis_f64[i][j] = basis[i][j] as f64;
        }
    }

    // Compute GS mu coefficients
    let mut mu_mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        mu_mat[i][i] = 1.0;
        for j in 0..i {
            let num = inner_prod(&basis_f64[i], &ortho[j], w);
            mu_mat[i][j] = num / b_star_norms[j];
        }
    }

    // Project target vector into the GS basis (theta)
    let mut theta = vec![0.0; n];
    for j in 0..n {
        let num = inner_prod(&v_f64, &ortho[j], w);
        theta[j] = num / b_star_norms[j];
    }

    // The component of v outside the span of the lattice is shared by every lattice point,
    // so it starts the partial distance stack.
    let mut outside = v_f64.clone();
    for j in 0..n {
        vec_sub_assign(&mut outside, &ortho[j], theta[j]);
    }
    let outside_dist = quad_form(&outside, w).max(0.0);

    // State setup
    let mut bound = with_tolerance(radius_sq);

    let mut x = vec![0i64; n];
    let mut c = vec![0.0; n];
    let mut p = vec![0.0; n + 1];
    let mut d = vec![0i64; n];
    let mut step = vec![0i64; n];

    // partial[k] = sum of x[i] * basis[i] for i >= k, filled in on the way down
    let mut partial = vec![vec![0i64; m]; n + 1];
    let mut point = vec![0i64; m];

    // Initialize the root node at level n - 1
    let mut k = n - 1;
    c[k] = theta[k];
    x[k] = c[k].round_ties_even() as i64;
    let y = c[k] - x[k] as f64;
    step[k] = if y >= 0.0 { 1 } else { -1 };
    d[k] = 1;
    p[n] = outside_dist;
    p[k] = p[k + 1] + y * y * b_star_norms[k];

    // Depth-first search
    loop {
        if p[k] <= bound {
            if k == 0 {
                // Reached a leaf node (a complete lattice point)
                for j in 0..m {
                    point[j] = x[0]
                        .checked_mul(basis[0][j])
                        .and_then(|t| t.checked_add(partial[1][j]))
                        .ok_or(DiophantineError::Overflow("enumerate: lattice vector"))?;
                }
                let r = visit(&x, &point, p[0])?;
                if r.is_nan() || r < 0.0 {
                    return Ok(());
                }
                bound = with_tolerance(r);
            } else {
                // Internal node: step down to level k - 1
                for j in 0..m {
                    partial[k][j] = x[k]
                        .checked_mul(basis[k][j])
                        .and_then(|t| t.checked_add(partial[k + 1][j]))
                        .ok_or(DiophantineError::Overflow("enumerate: lattice vector"))?;
                }

                k -= 1;
                let mut sum = 0.0;
                for i in (k + 1)..n {
                    sum += x[i] as f64 * mu_mat[i][k];
                }
                c[k] = theta[k] - sum;
                x[k] = c[k].round_ties_even() as i64;

                let y = c[k] - x[k] as f64;
                step[k] = if y >= 0.0 { 1 } else { -1 };
                d[k] = 1;
                p[k] = p[k + 1] + y * y * b_star_norms[k];
                continue;
            }
        } else {
            // Prune current branch: step back up to level k + 1
            k += 1;
            if k == n {
                return Ok(());
            }
        }

        // Advance to the next integer coefficient at level k, in alternating SE order
        x[k] += step[k] * d[k];
        step[k] = -step[k];
        d[k] += 1;
        let y = c[k] - x[k] as f64;
        p[k] = p[k + 1] + y * y * b_star_norms[k];
    }
}

/// The `k` best vectors found so far, best first, ordered by score and then by the vector.
struct TopK<S> {
    k: usize,
    items: Vec<(S, Vec<i64>)>,
}

impl<S: PartialOrd + Copy> TopK<S> {
    fn new(k: usize) -> Self {
        TopK {
            k,
            items: Vec::with_capacity(k),
        }
    }

    fn insert(&mut self, x: &[i64], score: S) {
        // How a kept item compares to the new one
        let cmp = |(s, y): &(S, Vec<i64>)| {
            s.partial_cmp(&score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| y.as_slice().cmp(x))
        };
        if self.items.len() >= self.k && self.items.last().is_none_or(|last| cmp(last).is_le()) {
            return;
        }
        let pos = self.items.partition_point(|item| cmp(item).is_lt());
        self.items.insert(pos, (score, x.to_vec()));
        self.items.truncate(self.k);
    }

    /// The squared radius that still holds every candidate for the list, computed from the
    /// worst kept score by `radius_sq`. Infinite until the list is full.
    fn radius_sq(&self, radius_sq: impl Fn(S) -> f64) -> f64 {
        match self.items.last() {
            Some((s, _)) if self.items.len() >= self.k => radius_sq(*s),
            _ => f64::INFINITY,
        }
    }

    fn into_vecs(self) -> Matrix<i64> {
        self.items.into_iter().map(|(_, x)| x).collect()
    }
}

/// Exact Closest Vector Problem (CVP) using Schnorr-Euchner enumeration.
///
/// Returns the exact closest vector in the lattice to the target vector `v`.
/// Ties are broken by the vector itself (lexicographically smallest).
/// For reasonable performance, `basis` MUST be highly reduced (e.g., LLL or BKZ) before calling.
///
/// # Arguments
/// * `v` - The target vector (should match the number of columns in the basis).
/// * `basis` - The lattice basis (row vectors).
/// * `w` - The metric quadratic form matrix (weights).
pub fn cvp_exact(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
) -> Result<Vec<i64>, DiophantineError> {
    Ok(cvp_top_k(v, basis, w, 1)?.swap_remove(0))
}

/// The `k` vectors of the lattice closest to the target `v` under the quadratic form `w`,
/// closest first. Ties are broken by the vector itself (lexicographically smallest first),
/// so the result does not depend on the choice of basis.
///
/// Returns fewer than `k` vectors only if the basis is empty (the lattice is just the origin).
/// For reasonable performance, `basis` should be reduced (e.g. LLL) under `w`, and `k` small.
///
/// # Arguments
/// * `v` - The target vector (should match the number of columns in the basis).
/// * `basis` - The lattice basis (row vectors), linearly independent.
/// * `w` - The metric quadratic form matrix (weights), definite on the span of `basis`.
/// * `k` - The number of vectors to return.
pub fn cvp_top_k(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
    k: usize,
) -> Result<Matrix<i64>, DiophantineError> {
    check_dims(v, basis, w)?;
    if k == 0 {
        return Ok(vec![]);
    }

    let mut top = TopK::new(k);
    let mut diff = vec![0.0; v.len()];
    enumerate(v, basis, w, f64::INFINITY, |_, x, _| {
        for j in 0..v.len() {
            let dj = v[j]
                .checked_sub(x[j])
                .ok_or(DiophantineError::Overflow("cvp_top_k: difference"))?;
            diff[j] = dj as f64;
        }
        top.insert(x, quad_form(&diff, w));
        Ok(top.radius_sq(|s| s))
    })?;

    Ok(top.into_vecs())
}

/// The `k` vectors `x` of the lattice for which `v - x` is smallest under the weighted
/// L1 norm `sum weights[i] * |(v - x)[i]|`, best first. Ties are broken by the weighted
/// L2 norm `sum (weights[i] * (v - x)[i])^2`, then by `x` itself (lexicographically smallest
/// first), so the result does not depend on the choice of basis.
///
/// Weights must be non-negative and may be zero, as long as no nonzero lattice vector has
/// zero weighted norm. For reasonable performance, `basis` should be reduced (e.g. LLL)
/// under `diag(weights^2)`, and `k` small.
///
/// Returns fewer than `k` vectors only if the basis is empty (the lattice is just the origin).
///
/// # Arguments
/// * `v` - The target vector (should match the number of columns in the basis).
/// * `basis` - The lattice basis (row vectors), linearly independent.
/// * `weights` - The weight of each coordinate.
/// * `k` - The number of vectors to return.
pub fn cvp_l1_top_k(
    v: &[i64],
    basis: &Matrix<i64>,
    weights: &[i64],
    k: usize,
) -> Result<Matrix<i64>, DiophantineError> {
    let m = v.len();
    if weights.len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "Weights should have same length as target vector".to_string(),
        ));
    }
    if weights.iter().any(|&wi| wi < 0) {
        return Err(DiophantineError::InvalidArgument(
            "Weights must be non-negative".to_string(),
        ));
    }

    // The weighted L2 norm is at most the weighted L1 norm, so every candidate at least as
    // good as the current k-th best lies in the L2 ball of that radius under diag(weights^2).
    let mut w = vec![vec![0.0; m]; m];
    for i in 0..m {
        w[i][i] = (weights[i] as f64).powi(2);
    }
    check_dims(v, basis, &w)?;
    if k == 0 {
        return Ok(vec![]);
    }

    let mut top = TopK::new(k);
    enumerate(v, basis, &w, f64::INFINITY, |_, x, _| {
        let mut l1: i64 = 0;
        let mut l2: i64 = 0;
        for j in 0..m {
            let wd = v[j]
                .checked_sub(x[j])
                .and_then(|dj| dj.checked_mul(weights[j]))
                .ok_or(DiophantineError::Overflow("cvp_l1_top_k: norm"))?;
            l1 = wd
                .checked_abs()
                .and_then(|a| l1.checked_add(a))
                .ok_or(DiophantineError::Overflow("cvp_l1_top_k: norm"))?;
            l2 = wd
                .checked_mul(wd)
                .and_then(|s| l2.checked_add(s))
                .ok_or(DiophantineError::Overflow("cvp_l1_top_k: norm"))?;
        }
        top.insert(x, (l1, l2));
        Ok(top.radius_sq(|(l1, _)| (l1 as f64).powi(2)))
    })?;

    Ok(top.into_vecs())
}

/// Exact Shortest Vector Problem (SVP) using Schnorr-Euchner enumeration.
///
/// Returns the exact shortest **non-zero** vector in the lattice.
/// Ties are broken by the vector itself (lexicographically smallest).
/// For reasonable performance, `basis` MUST be highly reduced (e.g., LLL or BKZ) before calling.
///
/// # Arguments
/// * `basis` - The lattice basis (row vectors).
/// * `w` - The metric quadratic form matrix (weights).
pub fn svp_exact(basis: &Matrix<i64>, w: &Matrix<f64>) -> Result<Vec<i64>, DiophantineError> {
    let Some(first) = basis.first() else {
        return Ok(vec![]);
    };
    let origin = vec![0; first.len()];

    let mut top = TopK::new(1);
    let mut x_f64 = vec![0.0; origin.len()];
    enumerate(&origin, basis, w, f64::INFINITY, |coeffs, x, _| {
        if coeffs.iter().any(|&c| c != 0) {
            for (xf, &xi) in x_f64.iter_mut().zip(x) {
                *xf = xi as f64;
            }
            top.insert(x, quad_form(&x_f64, w));
        }
        Ok(top.radius_sq(|s| s))
    })?;

    Ok(top.into_vecs().pop().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{eye, integer_det};

    fn norm_sq(v: &[i64]) -> i64 {
        v.iter().map(|x| x * x).sum()
    }

    #[test]
    fn lll_identity() {
        // LLL of identity is identity
        let basis = vec![vec![1, 0], vec![0, 1]];
        let w = eye(2);
        let reduced = lll(&basis, 0.99, &w).unwrap();
        assert_eq!(basis, reduced);
    }

    #[test]
    fn lll_eye() {
        // Should reduce to identity
        let basis = vec![
            vec![1, 0, 0, 0],
            vec![10, 1, 0, 0],
            vec![10, 0, 1, 0],
            vec![10, 0, 0, 1],
        ];
        let w = eye(4);

        let reduced = lll(&basis, 0.99, &w).unwrap();

        let det_in = integer_det(&basis).unwrap().abs();
        let det_out = integer_det(&reduced).unwrap().abs();
        assert_eq!(det_in, det_out, "LLL must preserve lattice determinant");

        assert!(
            reduced
                == vec![
                    vec![1, 0, 0, 0],
                    vec![0, 1, 0, 0],
                    vec![0, 0, 1, 0],
                    vec![0, 0, 0, 1],
                ]
        );
    }

    #[test]
    fn lll_det() {
        // Some random 4x4 matrix
        let basis = vec![
            vec![12, 10, 54, 46],
            vec![23, 23, -56, 23],
            vec![43, -8, 53, 20],
            vec![10, 8, -89, 1],
        ];
        let w = eye(4);

        let reduced = lll(&basis, 0.99, &w).unwrap();

        let det_in = integer_det(&basis).unwrap().abs();
        let det_out = integer_det(&reduced).unwrap().abs();

        assert_eq!(det_in, det_out, "LLL must preserve determinant");

        let max_input_norm = basis.iter().map(|r| norm_sq(r)).max().unwrap();
        let first_reduced_norm = norm_sq(&reduced[0]);

        assert!(
            first_reduced_norm < max_input_norm,
            "First vector should be reduced"
        );
    }

    #[test]
    fn phi_lll() {
        // Find an integer polynomial for the golden ratio
        // Last row:
        //   round(10_000 * phi^2)
        //   round(10_000 * phi)
        //   10_000
        //
        // First vector should be [1 -1 -1 0]
        // Since it is the root of x^2 - x - 1

        let basis = vec![
            vec![1, 0, 0, 26_180],
            vec![0, 1, 0, 16_180],
            vec![0, 0, 1, 10_000],
        ];

        let w = eye(4);
        let reduced = lll(&basis, 0.99, &w).unwrap();

        // Don't know what sign it is going to give
        assert!(reduced[0] == vec![1, -1, -1, 0] || reduced[0] == vec![-1, 1, 1, 0]);
    }

    #[test]
    fn phi_exact() {
        let basis = vec![
            vec![1, 0, 0, 26_180],
            vec![0, 1, 0, 16_180],
            vec![0, 0, 1, 10_000],
        ];

        let w = eye(4);
        let sv = svp_exact(&basis, &w).unwrap();
        assert!(sv == vec![1, -1, -1, 0] || sv == vec![-1, 1, 1, 0]);
    }

    #[test]
    fn nearest_plane_simple_grid() {
        // If the lattice is Z^2, this should always return the same vector
        let basis = vec![vec![1, 0], vec![0, 1]];
        let w = eye(2);

        let target = vec![10, 10];
        let res = nearest_plane(&target, &basis, &w).unwrap();
        assert_eq!(res, vec![10, 10]);

        let target = vec![123, 456];
        let res = nearest_plane(&target, &basis, &w).unwrap();
        assert_eq!(res, vec![123, 456]);
    }

    #[test]
    fn nearest_plane_scaled_lattice() {
        // Lattice 2*Z^2
        let basis = vec![vec![2, 0], vec![0, 2]];
        let w = eye(2);

        let target = vec![3, 3];
        let res = nearest_plane(&target, &basis, &w).unwrap();
        assert_eq!(res, vec![4, 4]);

        let target = vec![5, 1];
        let res = nearest_plane(&target, &basis, &w).unwrap();
        assert_eq!(res, vec![4, 0]);
    }

    #[test]
    fn test_lll_dims() {
        let basis = vec![vec![1, 0], vec![0, 1], vec![0, 1]];
        let w = eye(3);
        let res = lll(&basis, 0.75, &w);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));

        let basis = eye(3);
        let w = vec![vec![1., 0.], vec![0., 1.], vec![0., 1.]];
        let res = lll(&basis, 0.75, &w);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));
    }

    #[test]
    fn test_nearest_plane_dims() {
        let target = vec![1, 2, 3, 4];
        let basis = eye(3);
        let w = eye(3);
        let res = nearest_plane(&target, &basis, &w);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));

        let target = vec![1, 2, 3];
        let basis = eye(3);
        let w = eye(4);
        let res = nearest_plane(&target, &basis, &w);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));
    }

    #[test]
    fn svp_identity() {
        let basis = vec![vec![1, 0], vec![0, 1]];
        let w = eye(2);
        let sv = svp_exact(&basis, &w).unwrap();

        let norm = norm_sq(&sv);
        assert_eq!(norm, 1, "Shortest vector in Z^2 should have norm 1");
        assert!(sv == vec![1, 0] || sv == vec![0, 1] || sv == vec![-1, 0] || sv == vec![0, -1]);
    }

    #[test]
    fn svp_known_lattice() {
        let basis = vec![vec![1, 13, 14], vec![0, 12, 13]];
        let w = eye(3);

        let sv = svp_exact(&basis, &w).unwrap();

        // Should be [1, 1, 1]
        assert_eq!(norm_sq(&sv), 3);
    }

    #[test]
    fn test_cvp_exact_in_lattice() {
        // If the target is exactly a lattice point, the distance should be 0
        let basis = vec![vec![2, 0], vec![0, 2]];
        let w = eye(2);
        let target = vec![4, 6];

        let closest = cvp_exact(&target, &basis, &w).unwrap();
        assert_eq!(closest, vec![4, 6]);
    }

    #[test]
    fn test_cvp_exact_halfway() {
        let basis = vec![vec![2, 0], vec![0, 2]];
        let w = eye(2);

        // Target is directly in the middle of a 2x2 square cell [2, 0] to [4, 2]
        let target = vec![3, 1];
        let closest = cvp_exact(&target, &basis, &w).unwrap();

        // Distance from [3, 1] to any corner of its cell ([2,0], [4,0], [2,2], [4,2]) is exactly 2.
        let dist = norm_sq(&[closest[0] - target[0], closest[1] - target[1]]);
        assert_eq!(dist, 2);
    }

    #[test]
    fn cvp_top_k_ties() {
        // The four corners of the cell are equally close, and come out ordered by vector
        let basis = vec![vec![2, 0], vec![0, 2]];
        let target = vec![1, 1];
        let expected = vec![vec![0, 0], vec![0, 2], vec![2, 0], vec![2, 2]];

        assert_eq!(cvp_top_k(&target, &basis, &eye(2), 4).unwrap(), expected);
        assert_eq!(cvp_l1_top_k(&target, &basis, &[1, 1], 4).unwrap(), expected);
        assert_eq!(cvp_exact(&target, &basis, &eye(2)).unwrap(), vec![0, 0]);
    }

    #[test]
    fn cvp_l1_differs_from_l2() {
        // From (3, 0), the lattice point (0, 0) is off by (3, 0) and (1, -2) by (2, 2):
        // L1 distances 3 and 4, but L2 distances 9 and 8.
        let basis = vec![vec![1, -2]];
        let target = vec![3, 0];
        let res = cvp_l1_top_k(&target, &basis, &[1, 1], 2).unwrap();
        assert_eq!(res, vec![vec![0, 0], vec![1, -2]]);
        let res = cvp_top_k(&target, &basis, &eye(2), 2).unwrap();
        assert_eq!(res, vec![vec![1, -2], vec![0, 0]]);
    }

    #[test]
    fn cvp_top_k_outside_span() {
        // Lattice spanned by (1, 0, 0) and (0, 1, 0), target off the plane
        let basis = vec![vec![1, 0, 0], vec![0, 1, 0]];
        let target = vec![3, -2, 7];
        let res = cvp_top_k(&target, &basis, &eye(3), 5).unwrap();
        assert_eq!(res[0], vec![3, -2, 0]);
        assert_eq!(res.len(), 5);
        let res = cvp_l1_top_k(&target, &basis, &[1, 1, 1], 5).unwrap();
        assert_eq!(res[0], vec![3, -2, 0]);
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn cvp_l1_zero_weight() {
        // Vectors (a, a + b, b): only zero has zero weight under (0, 1, 2)
        let basis = vec![vec![1, 1, 0], vec![0, 1, 1]];
        let target = vec![10, 3, 1];
        let res = cvp_l1_top_k(&target, &basis, &[0, 1, 2], 3).unwrap();
        // (2, 3, 1) is a lattice point with weighted distance 0
        assert_eq!(res[0], vec![2, 3, 1]);
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn top_k_edge_cases() {
        let basis = vec![vec![1, 0], vec![0, 1]];
        let target = vec![1, 2];
        assert!(cvp_top_k(&target, &basis, &eye(2), 0).unwrap().is_empty());
        assert!(
            cvp_l1_top_k(&target, &basis, &[1, 1], 0)
                .unwrap()
                .is_empty()
        );

        // Rank 0: the lattice is just the origin
        let empty: Matrix<i64> = vec![];
        assert_eq!(
            cvp_top_k(&target, &empty, &eye(2), 3).unwrap(),
            vec![vec![0, 0]]
        );
        assert_eq!(
            cvp_l1_top_k(&target, &empty, &[1, 1], 3).unwrap(),
            vec![vec![0, 0]]
        );
        assert_eq!(cvp_exact(&target, &empty, &eye(2)).unwrap(), vec![0, 0]);

        // Rank 1, many more points than a small box
        let line = vec![vec![1, 1]];
        let res = cvp_top_k(&[0, 0], &line, &eye(2), 7).unwrap();
        assert_eq!(res.len(), 7);
        assert_eq!(res[0], vec![0, 0]);
        assert_eq!(res[5], vec![-3, -3]);
        assert_eq!(res[6], vec![3, 3]);

        let res = cvp_l1_top_k(&target, &basis, &[1, -1], 1);
        assert!(matches!(res, Err(DiophantineError::InvalidArgument(_))));
        let res = cvp_l1_top_k(&target, &basis, &[1, 1, 1], 1);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));
    }

    #[test]
    fn svp_cvp_dims() {
        let basis = vec![vec![1, 0], vec![0, 1], vec![0, 1]];
        let w = eye(3);
        let res = svp_exact(&basis, &w);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));

        let target = vec![1, 2, 3, 4];
        let basis = eye(3);
        let w = eye(3);
        let res = cvp_exact(&target, &basis, &w);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::{eye, integer_det};
    use proptest::prelude::*;

    fn norm_sq(v: &[i64]) -> i64 {
        v.iter().map(|x| x * x).sum()
    }

    /// Whether `x` is an integer combination of the rows of the square, nonsingular `basis`.
    ///
    /// By Cramer's rule the coefficients are `det(B_i) / det(B)`, where `B_i` is `basis` with
    /// row `i` replaced by `x`. Exact, and unlike HNF it has no coefficient growth.
    fn in_lattice(x: &[i64], basis: &Matrix<i64>) -> bool {
        let det = integer_det(basis).unwrap();
        (0..basis.len()).all(|i| {
            let mut b_i = basis.clone();
            b_i[i] = x.to_vec();
            integer_det(&b_i).unwrap() % det == 0
        })
    }

    fn matrix(rows: usize, cols: usize, max_val: i64) -> impl Strategy<Value = Matrix<i64>> {
        proptest::collection::vec(proptest::collection::vec(-max_val..max_val, cols), rows)
    }

    prop_compose! {
        fn random_basis()(n in 2usize..=5)(mat in matrix(n, n, 30)) -> Matrix<i64> {
            mat
        }
    }

    prop_compose! {
        fn random_basis_target()(n in 2usize..=5)
        (
            basis in matrix(n, n, 20),
            target in proptest::collection::vec(-100i64..100, n),
        ) -> (Matrix<i64>, Vec<i64>) {
            (basis, target)
        }
    }

    /// B diag(q) B^T, exact
    fn weighted_gram(basis: &Matrix<i64>, q: &[i64]) -> Matrix<i64> {
        basis
            .iter()
            .map(|a| {
                basis
                    .iter()
                    .map(|b| (0..q.len()).map(|j| a[j] * q[j] * b[j]).sum())
                    .collect()
            })
            .collect()
    }

    fn combine(coeffs: &[i64], basis: &Matrix<i64>, m: usize) -> Vec<i64> {
        let mut x = vec![0; m];
        for (c, row) in coeffs.iter().zip(basis) {
            for j in 0..m {
                x[j] += c * row[j];
            }
        }
        x
    }

    /// The top `k` of the lattice points with coefficients within `r` of `center`, together
    /// with `extra`, ordered by `score` and then by the vector.
    fn bruteforce_top_k<S: Ord>(
        basis: &Matrix<i64>,
        m: usize,
        center: &[i64],
        r: i64,
        extra: &Matrix<i64>,
        k: usize,
        score: impl Fn(&[i64]) -> S,
    ) -> Matrix<i64> {
        let n = basis.len();
        let mut points = extra.clone();
        let mut coeffs: Vec<i64> = center.iter().map(|c| c - r).collect();
        loop {
            points.push(combine(&coeffs, basis, m));
            let Some(i) = (0..n).find(|&i| coeffs[i] < center[i] + r) else {
                break;
            };
            coeffs[i] += 1;
            for c in 0..i {
                coeffs[c] = center[c] - r;
            }
        }
        points.sort_by(|a, b| score(a).cmp(&score(b)).then_with(|| a.cmp(b)));
        points.dedup();
        points.truncate(k);
        points
    }

    /// A basis of `n <= m` independent rows, the coefficients of a lattice point near the
    /// target, the target, diagonal weights (possibly zero) definite on the span, and `k`.
    fn cvp_case() -> impl Strategy<Value = (Matrix<i64>, Vec<i64>, Vec<i64>, Vec<i64>, usize)> {
        (1usize..=4)
            .prop_flat_map(|m| (1..=m, Just(m)))
            .prop_flat_map(|(n, m)| {
                // A zero weight can only be definite on the span if the span is not everything
                let min_weight = if n < m { 0i64 } else { 1 };
                (
                    matrix(n, m, 6),
                    proptest::collection::vec(-3i64..=3, n),
                    proptest::collection::vec(-4i64..=4, m),
                    proptest::collection::vec(min_weight..=3, m),
                    1usize..=6,
                )
            })
            .prop_filter(
                "form must be definite on the span",
                |(basis, _, _, weights, _)| {
                    let q: Vec<i64> = weights.iter().map(|w| w * w).collect();
                    integer_det(&weighted_gram(basis, &q)).unwrap_or(0) != 0
                },
            )
            .prop_map(|(basis, center, noise, weights, k)| {
                let m = noise.len();
                let target: Vec<i64> = combine(&center, &basis, m)
                    .iter()
                    .zip(&noise)
                    .map(|(x, e)| x + e)
                    .collect();
                (basis, center, target, weights, k)
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn test_cvp_top_k_brute_force((basis, center, target, weights, k) in cvp_case()) {
            let m = target.len();
            let q: Vec<i64> = weights.iter().map(|w| w * w).collect();

            let mut w = vec![vec![0.0; m]; m];
            for i in 0..m {
                w[i][i] = q[i] as f64;
            }
            let l2 = |x: &[i64]| -> i64 {
                (0..m).map(|j| q[j] * (target[j] - x[j]).pow(2)).sum()
            };
            let l1 = |x: &[i64]| -> (i64, i64) {
                let a = (0..m).map(|j| weights[j] * (target[j] - x[j]).abs()).sum();
                (a, l2(x))
            };

            let reduced = lll(&basis, 0.99, &w).unwrap();

            for b in [&basis, &reduced] {
                let res = cvp_top_k(&target, b, &w, k).unwrap();
                let brute = bruteforce_top_k(&basis, m, &center, 4, &res, k, l2);
                prop_assert_eq!(&res, &brute, "L2 top-k differs from brute force");

                let res = cvp_l1_top_k(&target, b, &weights, k).unwrap();
                let brute = bruteforce_top_k(&basis, m, &center, 4, &res, k, l1);
                prop_assert_eq!(&res, &brute, "L1 top-k differs from brute force");
            }
        }

        #[test]
        fn test_cvp_top_k_basis_independent((basis, target) in random_basis_target(), k in 1usize..=8) {
            prop_assume!(integer_det(&basis).unwrap_or(0) != 0);
            let n = basis.len();
            let w = eye(n);
            let weights = vec![1; n];
            let reduced = lll(&basis, 0.99, &w).unwrap();

            prop_assert_eq!(
                cvp_top_k(&target, &basis, &w, k).unwrap(),
                cvp_top_k(&target, &reduced, &w, k).unwrap()
            );
            prop_assert_eq!(
                cvp_l1_top_k(&target, &basis, &weights, k).unwrap(),
                cvp_l1_top_k(&target, &reduced, &weights, k).unwrap()
            );
            prop_assert_eq!(
                cvp_top_k(&target, &reduced, &w, 1).unwrap()[0].clone(),
                cvp_exact(&target, &reduced, &w).unwrap()
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        #[test]
        fn test_lll_properties(
            basis in random_basis()
        ) {
            // Skip singular matrices
            let det_orig = integer_det(&basis).unwrap_or(0);
            prop_assume!(det_orig != 0);

            let n = basis.len();

            let w = eye(n);
            let delta = 0.75;
            let reduced = lll(&basis, delta, &w).unwrap();

            // Determinant is preserved
            let det_red = integer_det(&reduced).unwrap();
            prop_assert_eq!(det_orig.abs(), det_red.abs(), "Determinant magnitude changed!");

            let ortho = gramschmidt(&reduced, &w);
            let n = reduced.len();

            // Size Reduction Condition: |mu_{i,j}| <= 0.5
            for i in 0..n {
                for j in 0..i {
                    let mu_ij = mu(&reduced, &ortho, &w, i, j);
                    // Add tiny epsilon for f64 math
                    prop_assert!(
                        mu_ij.abs() <= 0.500001,
                        "Size reduction failed: mu_{},{} = {}", i, j, mu_ij
                    );
                }
            }

            // Lovasz Condition: delta * ||b*_{k-1}||^2 <= ||b*_k||^2 + mu_{k, k-1}^2 ||b*_{k-1}||^2
            for k in 1..n {
                let mu_k_k1 = mu(&reduced, &ortho, &w, k, k - 1);
                let norm_ortho_k1 = inner_prod(&ortho[k - 1], &ortho[k - 1], &w);
                let norm_ortho_k = inner_prod(&ortho[k], &ortho[k], &w);

                let lhs = delta * norm_ortho_k1;
                let rhs = norm_ortho_k + mu_k_k1.powi(2) * norm_ortho_k1;

                prop_assert!(
                    lhs <= rhs + 1e-6,
                    "Lovasz condition failed at step {}: {} > {}", k, lhs, rhs
                );
            }
        }

        #[test]
        fn test_nearest_plane_properties((basis, target) in random_basis_target()) {
            prop_assume!(integer_det(&basis).unwrap_or(0) != 0);

            let n = basis.len();
            let w = eye(n);

            // Nearest plane requires an LLL-reduced basis to work effectively
            let reduced = lll(&basis, 0.75, &w).unwrap();
            let np = nearest_plane(&target, &reduced, &w).unwrap();

            // The result must be a point on the lattice.
            prop_assert!(in_lattice(&np, &reduced), "nearest_plane result is not an integer combination of the basis!");

            // The error vector must fall within the fundamental parallelepiped
            // of the Gram-Schmidt basis.
            let ortho = gramschmidt(&reduced, &w);
            let error: Vec<f64> = target.iter().zip(np.iter()).map(|(&t, &p)| (t - p) as f64).collect();

            for i in 0..n {
                let a = &ortho[i];
                let num = inner_prod(a, &error, &w);
                let den = inner_prod(a, a, &w);
                let mu_err = if den.abs() < 1e-9 { 0.0 } else { num / den };

                prop_assert!(
                    mu_err.abs() <= 0.500001,
                    "Error vector projection onto GS vector {} exceeds 0.5: {}", i, mu_err
                );
            }
        }
        #[test]
        fn test_svp_exact_properties(basis in random_basis()) {
            let det_orig = integer_det(&basis).unwrap_or(0);
            prop_assume!(det_orig != 0);

            let n = basis.len();
            let w = eye(n);

            // LLL-reduce first
            let reduced = lll(&basis, 0.99, &w).unwrap();
            let svp_res = svp_exact(&reduced, &w).unwrap();

            // Result must be non-zero
            prop_assert!(svp_res.iter().any(|&x| x != 0), "SVP exact returned the zero vector!");

            let svp_norm = norm_sq(&svp_res);
            let lll_first_norm = norm_sq(&reduced[0]);

            // SVP must find a vector at least as short as LLL's best approximation
            prop_assert!(
                svp_norm <= lll_first_norm,
                "SVP exact found a longer vector ({}) than LLL ({})", svp_norm, lll_first_norm
            );

            // Must be a valid lattice point
            prop_assert!(in_lattice(&svp_res, &reduced), "SVP exact result is not in the lattice!");

            // No nonzero point with small coefficients is shorter
            let box_min = bruteforce_top_k(&reduced, n, &vec![0; n], 2, &vec![], 2, norm_sq)
                .iter()
                .map(|x| norm_sq(x))
                .find(|&s| s != 0)
                .unwrap();
            prop_assert!(
                svp_norm <= box_min,
                "SVP exact ({}) is longer than a small combination ({})", svp_norm, box_min
            );
        }

        #[test]
        fn test_cvp_exact_properties((basis, target) in random_basis_target()) {
            let det_orig = integer_det(&basis).unwrap_or(0);
            prop_assume!(det_orig != 0);

            let n = basis.len();
            let w = eye(n);

            // Perform CVP on a reduced basis
            let reduced = lll(&basis, 0.99, &w).unwrap();

            let cvp_res = cvp_exact(&target, &reduced, &w).unwrap();
            let babai_res = nearest_plane(&target, &reduced, &w).unwrap();

            let error_cvp: Vec<i64> = target.iter().zip(cvp_res.iter()).map(|(&t, &c)| t - c).collect();
            let error_babai: Vec<i64> = target.iter().zip(babai_res.iter()).map(|(&t, &c)| t - c).collect();

            let dist_cvp = norm_sq(&error_cvp);
            let dist_babai = norm_sq(&error_babai);

            // Exact CVP must always yield a distance <= the approximate Babai's nearest plane distance
            prop_assert!(
                dist_cvp <= dist_babai,
                "Exact CVP yielded worse distance {} than Babai's approximate {}", dist_cvp, dist_babai
            );

            // Must be a valid lattice point (Linear combination of the reduced basis)
            prop_assert!(in_lattice(&cvp_res, &reduced), "CVP exact result is not in the lattice!");
        }
    }
}
