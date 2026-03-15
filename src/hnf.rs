use crate::Matrix;

/// Computes the Hermite Normal Form of an integer matrix.
/// Returns the HNF basis.
pub fn hnf(basis: &Matrix<i64>) -> Result<Matrix<i64>, String> {
    // TODO: We do some extra work here for U which can be avoided
    let (h, _) = extended_hnf(basis)?;
    Ok(h)
}

/// Computes the Extended Hermite Normal Form of an integer matrix.
/// Returns a tuple `(H, U)` where `H` is the HNF and `U` is the unimodular
/// transformation matrix such that `U * A = H`.
pub fn extended_hnf(basis: &Matrix<i64>) -> Result<(Matrix<i64>, Matrix<i64>), String> {
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
                                        .ok_or_else(|| "Overflow in HNF.".to_string())?;
                                }
                                for col in 0..n {
                                    u[i][col] = k
                                        .checked_mul(u[si][col])
                                        .and_then(|val| u[i][col].checked_sub(val))
                                        .ok_or_else(|| "Overflow in HNF.".to_string())?;
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
}
