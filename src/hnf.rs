//! Row-style Hermite Normal Form (HNF) calculation.

use crate::error::DiophantineError;
use crate::solve_diophantine;
use crate::integer_det;
use crate::util::transpose;
use crate::Matrix;

/// Computes the Hermite Normal Form of an integer matrix.
///
/// Returns the HNF basis.
pub fn hnf(basis: &Matrix<i64>) -> Result<Matrix<i64>, DiophantineError> {
    // TODO: We do some extra work here for U which can be avoided
    let (h, _) = extended_hnf(basis)?;
    Ok(h)
}

/// Computes the Extended Hermite Normal Form of an integer matrix.
///
/// Returns a tuple `(H, U)` where `H` is the HNF and `U` is the unimodular
/// transformation matrix such that `U * A = H`.
pub fn extended_hnf(basis: &Matrix<i64>) -> Result<(Matrix<i64>, Matrix<i64>), DiophantineError> {
    let mut a = basis.clone();
    let n = a.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    let m = a[0].len();

    // U starts as the identity matrix
    let mut u = vec![vec![0; n]; n];
    for i in 0..n {
        u[i][i] = 1;
    }

    let mut si = 0;
    let mut sj = 0;

    while si < n && sj < m {
        let pivot_idx = (si..n)
            .filter(|&i| a[i][sj] != 0)
            .min_by_key(|&i| a[i][sj].abs());

        match pivot_idx {
            None => {
                sj += 1;
            }
            Some(row) => {
                if row != si {
                    a.swap(si, row);
                    u.swap(si, row);
                }

                // Eliminate entries below pivot (Euclidean step)
                for i in (si + 1)..n {
                    if a[i][sj] != 0 {
                        let k = a[i][sj] / a[si][sj];
                        for col in 0..m {
                            a[i][col] -= k * a[si][col];
                        }
                        for col in 0..n {
                            u[i][col] -= k * u[si][col];
                        }
                    }
                }

                // Check if column is cleared below pivot
                let row_done = ((si + 1)..n).all(|i| a[i][sj] == 0);

                if row_done {
                    // Ensure pivot is positive
                    if a[si][sj].is_negative() {
                        for col in 0..m {
                            a[si][col] = -a[si][col];
                        }
                        for col in 0..n {
                            u[si][col] = -u[si][col];
                        }
                    }

                    // Eliminate entries above pivot
                    if a[si][sj] != 0 {
                        for i in 0..si {
                            let k = a[i][sj].div_euclid(a[si][sj]);
                            if k != 0 {
                                for col in 0..m {
                                    a[i][col] = k
                                        .checked_mul(a[si][col])
                                        .and_then(|val| a[i][col].checked_sub(val))
                                        .ok_or(DiophantineError::Overflow("Overflow in HNF"))?;
                                }
                                for col in 0..n {
                                    u[i][col] = k
                                        .checked_mul(u[si][col])
                                        .and_then(|val| u[i][col].checked_sub(val))
                                        .ok_or(DiophantineError::Overflow("Overflow in HNF"))?;
                                }
                            }
                        }
                    }

                    si += 1;
                    sj += 1;
                }
            }
        }
    }

    Ok((a, u))
}

/// Computes the saturation of the lattice spanned by the columns of a matrix.
///
/// Returns the Hermite Normal Form of the saturated basis.
///
/// This process "saturates" the lattice, effectively "filling in the gaps".
/// Mathematically, saturation finds a basis for all integer vectors that can be
/// expressed as a rational linear combination of the columns of the input matrix `m`.
/// It computes a basis for the intersection of the rational span of the columns
/// with the integer grid Z^n.
///
/// This is useful for finding the most "primitive" integer basis that spans the same
/// rational vector space as the input matrix. It has the effect of removing the
/// torsion part of the cokernel of the linear map represented by `m`.
///
/// # Example
/// Consider the matrix `M = [[2, 2], [0, 2]]`. The lattice it generates consists
/// of integer vectors `(2x+2y, 2y)`. The rational span of its columns is all of Q^2.
/// The integer vectors within this full rational span are simply Z^2. Therefore, the
/// saturation is the basis for Z^2, which is the identity matrix `[[1, 0], [0, 1]]`.
///
/// # Reference
/// Clément Pernet and William Stein.
///
/// Fast Computation of HNF of Random Integer Matrices (section 8).
/// Journal of Number Theory.
///
/// <https://doi.org/10.1016/j.jnt.2010.01.017>
pub fn saturation(m: &Matrix<i64>) -> Result<Matrix<i64>, DiophantineError> {
    if m.is_empty() {
        return Ok(vec![]);
    }
    let r = m.len();

    // Compute K, the HNF basis of the column space of M.
    let mt = transpose(m);
    let hnf_mt = hnf(&mt)?;
    // We only need the first `r` rows, as the image can have at most rank `r`.
    let k_t = hnf_mt.iter().take(r).cloned().collect();
    let k = transpose(&k_t);

    // If det(K) is 1, the map is already surjective onto its image basis.
    // The standard HNF is sufficient.
    if let Ok(det) = integer_det(&k)
        && det.abs() == 1
    {
        return Ok(hnf(m)?);
    }

    // Project M onto the basis K by solving KD = M for D.
    // This is the saturation step.
    let d = solve_diophantine(&k, m)?;

    // The result is the HNF of this saturated matrix D.
    Ok(hnf(&d)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integer_det;
    use crate::util::matmul;

    #[test]
    fn hnf_simple() {
        let m = vec![vec![2, 1], vec![0, 3]];
        let h = hnf(&m).unwrap();
        assert_eq!(h, vec![vec![2, 1], vec![0, 3]]);
    }

    #[test]
    fn hnf_reduction() {
        // [2, 0]
        // [4, 2]
        // Row 1 can remove 2*Row0. Result should be:
        // [2, 0]
        // [0, 2]
        let m = vec![vec![2, 0], vec![4, 2]];
        let h = hnf(&m).unwrap();
        assert_eq!(h, vec![vec![2, 0], vec![0, 2]]);
    }

    #[test]
    fn hnf_preserves_det() {
        let m = vec![vec![8, 2], vec![12, 4]];
        let h = hnf(&m).unwrap();

        let d_in = integer_det(&m).unwrap().abs();
        let d_out = integer_det(&h).unwrap().abs();

        assert_eq!(d_in, d_out);

        // Additionally check upper triangular structure
        assert_eq!(h[1][0], 0);
    }

    #[test]
    fn extended_hnf_unimodularity() {
        let a = vec![vec![8, 2], vec![12, 4]];
        let (h, u) = extended_hnf(&a).unwrap();

        // Determinant of U must be +/-1 to preserve the Z-lattice
        let det_u = integer_det(&u).unwrap().abs();
        assert_eq!(det_u, 1, "Transformation matrix must be unimodular");

        // Verify U * A = H
        let ua = matmul(&u, &a).unwrap();
        assert_eq!(ua, h);
    }

    #[test]
    fn test_saturation() {
        // This matrix represents a map from Z^2 -> Z^2 whose image is the
        // lattice spanned by [2,0] and [0,2], but M itself is not in HNF.
        // M = [[2, 2],
        //      [0, 2]]
        let m = vec![vec![2, 2], vec![0, 2]];

        // The standard HNF is [[2, 0], [0, 2]].
        let standard_hnf = hnf(&m).unwrap();
        assert_eq!(standard_hnf, vec![vec![2, 0], vec![0, 2]]);

        // The saturated HNF should be the identity matrix, since
        // the rational span is Q^2, and the intersection with Z^2
        // is Z^2 itself.
        let saturated = saturation(&m).unwrap();
        assert_eq!(saturated, vec![vec![1, 0], vec![0, 1]]);
    }

}
