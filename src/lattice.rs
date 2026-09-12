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
    let m = basis[0].len();
    if n == 0 {
        return Ok(vec![]);
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
    let m = basis[0].len(); // number of cols
    if n == 0 {
        return Ok(vec![]);
    }

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

/// Internal helper for Schnorr-Euchner enumeration.
///
/// Returns the integer coefficients of the basis vectors that yield the closest point.
/// If `nonzero_only` is true, the zero vector is considered an invalid solution (useful for SVP).
fn schnorr_euchner(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
    nonzero_only: bool,
) -> Result<Vec<i64>, DiophantineError> {
    let n = basis.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let m = basis[0].len();

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
    let v_f64: Vec<f64> = v.iter().map(|&x| x as f64).collect();
    for j in 0..n {
        let num = inner_prod(&v_f64, &ortho[j], w);
        theta[j] = num / b_star_norms[j];
    }

    // State setup
    let mut best_dist = f64::INFINITY;
    let mut best_x = vec![0i64; n];

    let mut x = vec![0i64; n];
    let mut c = vec![0.0; n];
    let mut p = vec![0.0; n + 1];
    let mut d = vec![0i64; n];
    let mut step = vec![0i64; n];

    let mut k = (n - 1) as isize;

    // Initialize the root node at level n - 1
    let ku = k as usize;
    c[ku] = theta[ku];
    x[ku] = c[ku].round_ties_even() as i64;
    let y = c[ku] - x[ku] as f64;
    step[ku] = if y >= 0.0 { 1 } else { -1 };
    d[ku] = 1;
    p[ku] = p[ku + 1] + y * y * b_star_norms[ku];

    // Depth-first search
    while k < n as isize {
        let ku = k as usize;

        // If partial distance is strictly less than the best found distance, move deeper
        if p[ku] < best_dist {
            if k == 0 {
                // Reached a leaf node (a complete lattice point)
                let valid = if nonzero_only {
                    x.iter().any(|&xi| xi != 0)
                } else {
                    true
                };

                if valid {
                    best_dist = p[ku];
                    best_x.copy_from_slice(&x);
                }

                // Advance to the next integer coefficient at this leaf node
                x[ku] += step[ku] * d[ku];
                step[ku] = -step[ku];
                d[ku] += 1;
                let y = c[ku] - x[ku] as f64;
                p[ku] = p[ku + 1] + y * y * b_star_norms[ku];
            } else {
                // Internal node: step down to level k - 1
                k -= 1;
                let ku = k as usize;

                let mut sum = 0.0;
                for i in (ku + 1)..n {
                    sum += x[i] as f64 * mu_mat[i][ku];
                }
                c[ku] = theta[ku] - sum;
                x[ku] = c[ku].round_ties_even() as i64;

                let y = c[ku] - x[ku] as f64;
                step[ku] = if y >= 0.0 { 1 } else { -1 };
                d[ku] = 1;
                p[ku] = p[ku + 1] + y * y * b_star_norms[ku];
            }
        } else {
            // Prune current branch: step back up to level k + 1
            k += 1;
            if k < n as isize {
                let ku = k as usize;
                // Prepare the next alternating coefficient in SE order
                x[ku] += step[ku] * d[ku];
                step[ku] = -step[ku];
                d[ku] += 1;

                let y = c[ku] - x[ku] as f64;
                p[ku] = p[ku + 1] + y * y * b_star_norms[ku];
            }
        }
    }

    Ok(best_x)
}

/// Exact Closest Vector Problem (CVP) using Schnorr-Euchner enumeration.
///
/// Returns the exact closest vector in the lattice to the target vector `v`.
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
    let n = basis.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let m = basis[0].len();

    if v.len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "Target vector should have same length as basis columns".to_string(),
        ));
    }
    if w.len() != m || w[0].len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "W must be square and match basis columns".to_string(),
        ));
    }

    let best_x = schnorr_euchner(v, basis, w, false)?;

    // Linearly combine the basic vectors according to best_x coefficients
    let mut result = vec![0; m];
    for i in 0..n {
        if best_x[i] != 0 {
            for j in 0..m {
                result[j] += best_x[i] * basis[i][j];
            }
        }
    }

    Ok(result)
}

/// Exact Shortest Vector Problem (SVP) using Schnorr-Euchner enumeration.
///
/// Returns the exact shortest **non-zero** vector in the lattice.
/// For reasonable performance, `basis` MUST be highly reduced (e.g., LLL or BKZ) before calling.
///
/// # Arguments
/// * `basis` - The lattice basis (row vectors).
/// * `w` - The metric quadratic form matrix (weights).
pub fn svp_exact(basis: &Matrix<i64>, w: &Matrix<f64>) -> Result<Vec<i64>, DiophantineError> {
    let n = basis.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let m = basis[0].len();

    if w.len() != m || w[0].len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "W must be square and match basis columns".to_string(),
        ));
    }

    // SVP is precisely CVP centering around the origin with a strict constraint of non-zero coordinates
    let origin = vec![0; m];
    let best_x = schnorr_euchner(&origin, basis, w, true)?;

    let mut result = vec![0; m];
    for i in 0..n {
        if best_x[i] != 0 {
            for j in 0..m {
                result[j] += best_x[i] * basis[i][j];
            }
        }
    }

    Ok(result)
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
    use crate::{eye, integer_det, solve_diophantine, transpose};
    use proptest::prelude::*;

    fn norm_sq(v: &[i64]) -> i64 {
        v.iter().map(|x| x * x).sum()
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
            // We verify this by solving: (reduced^T) * X = (np^T)
            let red_t = transpose(&reduced);
            let mut np_t = vec![vec![0; 1]; n];
            for i in 0..n {
                np_t[i][0] = np[i];
            }

            let sol = solve_diophantine(&red_t, &np_t);
            prop_assert!(sol.is_ok(), "nearest_plane result is not an integer combination of the basis!");

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
            let red_t = transpose(&reduced);
            let mut svp_t = vec![vec![0; 1]; n];
            for i in 0..n {
                svp_t[i][0] = svp_res[i];
            }
            let sol = solve_diophantine(&red_t, &svp_t);
            prop_assert!(sol.is_ok(), "SVP exact result is not in the lattice!");
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
            let red_t = transpose(&reduced);
            let mut cvp_t = vec![vec![0; 1]; n];
            for i in 0..n {
                cvp_t[i][0] = cvp_res[i];
            }
            let sol = solve_diophantine(&red_t, &cvp_t);
            prop_assert!(sol.is_ok(), "CVP exact result is not in the lattice!");
        }
    }
}
