use crate::util::vec_sub_assign;
use crate::Matrix;

pub(crate) fn inner_prod(x: &[f64], y: &[f64], w: &Matrix<f64>) -> f64 {
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

/// Helper for LLL: Calculate mu coefficient
fn mu(basis: &Matrix<i64>, ortho: &Matrix<f64>, w: &Matrix<f64>, i: usize, j: usize) -> f64 {
    let a = &ortho[j];
    // Convert basis row to f64 on the fly for calculation
    let b: Vec<f64> = basis[i].iter().map(|&x| x as f64).collect();

    let num = inner_prod(a, &b, w);
    let den = inner_prod(a, a, w);

    if den.abs() < 1e-9 {
        0.0
    } else {
        num / den
    }
}

/// LLL Lattice Reduction
///
/// # Arguments
/// * `basis` - The lattice basis (row vectors).
/// * `delta` - The reduction parameter (typically 0.75 or 0.99).
/// * `w` - The quadratic form matrix (weights). Pass Identity matrix for standard Euclidean.
pub fn lll(basis: &Matrix<i64>, delta: f64, w: &Matrix<f64>) -> Matrix<i64> {
    let mut basis = basis.clone();
    let n = basis.len();
    if n == 0 {
        return basis;
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

    basis
}

/// Babai's Nearest Plane Algorithm for approximate CVP
pub fn nearest_plane(v: &[i64], basis: &Matrix<i64>, w: &Matrix<f64>) -> Vec<i64> {
    let mut b = v.to_vec();
    let n = basis.len(); // number of rows
    if n == 0 {
        return v.to_vec();
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
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integer_det;
    use crate::util::eye;

    fn norm_sq(v: &[i64]) -> i64 {
        v.iter().map(|x| x * x).sum()
    }

    #[test]
    fn lll_identity() {
        // LLL of identity is identity
        let basis = vec![vec![1, 0], vec![0, 1]];
        let w = eye(2);
        let reduced = lll(&basis, 0.99, &w);
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

        let reduced = lll(&basis, 0.99, &w);

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

        let reduced = lll(&basis, 0.99, &w);

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
    fn lll_phi() {
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
        let reduced = lll(&basis, 0.99, &w);

        // Don't know what sign it is going to give
        assert!(reduced[0] == vec![1, -1, -1, 0] || reduced[0] == vec![-1, 1, 1, 0]);
    }

    #[test]
    fn nearest_plane_simple_grid() {
        // If the lattice is Z^2, this should always return the same vector
        let basis = vec![vec![1, 0], vec![0, 1]];
        let w = eye(2);

        let target = vec![10, 10];
        let res = nearest_plane(&target, &basis, &w);
        assert_eq!(res, vec![10, 10]);

        let target = vec![123, 456];
        let res = nearest_plane(&target, &basis, &w);
        assert_eq!(res, vec![123, 456]);
    }

    #[test]
    fn nearest_plane_scaled_lattice() {
        // Lattice 2*Z^2
        let basis = vec![vec![2, 0], vec![0, 2]];
        let w = eye(2);

        let target = vec![3, 3];
        let res = nearest_plane(&target, &basis, &w);
        assert_eq!(res, vec![4, 4]);

        let target = vec![5, 1];
        let res = nearest_plane(&target, &basis, &w);
        assert_eq!(res, vec![4, 0]);
    }
}
