//! Tools for calculations over lattices, which mostly comes down to doing linear algebra over the integers.
//!
//! This crate is not intended for cryptographic applications. We use `i64` internally for all calculations, so use only for relatively small numbers and dimensions.
//! All internal overflows are caught using checked ops and returned as errors, so there should not be any issues with silently wrong results in release mode.

// This lint tends to reduce clarity for loops over matrices
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]
#![warn(clippy::cast_lossless)]

pub mod error;
pub mod hnf;
pub mod lll;
mod proptest;
pub mod util;

use crate::error::{DiophantineError, MatrixError, OverflowError};
use crate::hnf::extended_hnf;
pub use crate::hnf::hnf;
use crate::util::checked_matmul;
use crate::util::transpose;

/// Type alias for a Matrix (row-major)
pub type Matrix<T> = Vec<Vec<T>>;

// Helper to prevent overflow during Bareiss step
fn checked_bareiss_op(a: &Matrix<i64>, i: usize, j: usize, k: usize) -> Option<i64> {
    let term1 = a[j][k].checked_mul(a[i][i])?;
    let term2 = a[j][i].checked_mul(a[i][k])?;
    term1.checked_sub(term2)
}

/// Calculates the exact integer determinant of `basis`.
pub fn integer_det(basis: &Matrix<i64>) -> Result<i64, MatrixError> {
    let n = basis.len();
    if n == 0 {
        return Err(MatrixError::Empty);
    }
    if basis[0].len() != n {
        return Err(MatrixError::NotSquare);
    }

    let mut a = basis.clone();
    let mut sign = 1;
    let mut prev = 1;

    for i in 0..(n - 1) {
        if a[i][i] == 0 {
            // Find pivot
            let swap_idx = ((i + 1)..n).find(|&r| a[r][i] != 0);
            match swap_idx {
                Some(k) => {
                    a.swap(i, k);
                    sign *= -1;
                }
                None => return Ok(0), // Column is all zeros
            }
        }

        for j in (i + 1)..n {
            for k in (i + 1)..n {
                let d = checked_bareiss_op(&a, i, j, k);
                match d {
                    Some(val) => a[j][k] = val / prev,
                    None => return Err(MatrixError::Overflow),
                }
            }
        }
        prev = a[i][i];
    }

    Ok(sign * a[n - 1][n - 1])
}

/// Computes the integer inverse of a square matrix if it exists.
///
/// A matrix has an integer inverse if and only if it is square and unimodular (determinant is ±1).
pub fn integer_inverse(basis: &Matrix<i64>) -> Result<Matrix<i64>, MatrixError> {
    let n = basis.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if basis[0].len() != n {
        return Err(MatrixError::NotSquare);
    }

    // Create an augmented matrix [A | I]
    let mut a = vec![vec![0i64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = basis[i][j];
        }
        a[i][n + i] = 1;
    }

    let mut prev = 1;
    let mut sign = 1;

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        if a[i][i] == 0 {
            let swap_idx = ((i + 1)..n).find(|&r| a[r][i] != 0);
            match swap_idx {
                Some(r) => {
                    a.swap(i, r);
                    sign *= -1;
                }
                None => return Err(MatrixError::Singular),
            }
        }

        for j in 0..n {
            if i == j {
                continue;
            }
            for k in (i + 1)..(2 * n) {
                let term1 = a[i][i].checked_mul(a[j][k]);
                let term2 = a[j][i].checked_mul(a[i][k]);

                match (term1, term2) {
                    (Some(t1), Some(t2)) => match t1.checked_sub(t2) {
                        Some(diff) => a[j][k] = diff / prev, // Guaranteed to be exact division
                        None => return Err(MatrixError::Overflow),
                    },
                    _ => return Err(MatrixError::Overflow),
                }
            }
        }

        // Clear the column for all other rows
        for j in 0..n {
            if i != j {
                a[j][i] = 0;
            }
        }
        prev = a[i][i];
    }

    let det = sign * a[n - 1][n - 1];
    if det.abs() != 1 {
        return Err(MatrixError::NotUnimodular(det));
    }

    // Extract the exact inverse from the right half of the augmented matrix.
    // The Gauss-Jordan Bareiss algorithm scales the entire matrix such that
    // the final diagonal elements are equal to the final pivot.
    let final_diag = a[n - 1][n - 1];
    let mut inv = vec![vec![0; n]; n];

    for i in 0..n {
        for j in 0..n {
            inv[i][j] = a[i][n + j] / final_diag;
        }
    }

    Ok(inv)
}

/// Computes the left kernel of an integer matrix.
///
/// Returns a basis of row vectors `x` such that `x * A = 0`.
pub fn left_kernel(basis: &Matrix<i64>) -> Result<Matrix<i64>, OverflowError> {
    let (h, u) = extended_hnf(basis)?;
    let mut kernel = Vec::new();

    for (i, row) in h.iter().enumerate() {
        if row.iter().all(|&val| val == 0) {
            kernel.push(u[i].clone());
        }
    }

    Ok(kernel)
}

/// Computes the right kernel (nullspace) of an integer matrix.
///
/// Returns a basis of row vectors `x` such that `A * x = 0`.
pub fn right_kernel(basis: &Matrix<i64>) -> Result<Matrix<i64>, OverflowError> {
    let t = transpose(basis);
    let res = left_kernel(&t)?;
    Ok(transpose(&res))
}

/// Helper to compute the partial dot product of a column of H and a column of Y
fn checked_partial_col_dot(
    h: &Matrix<i64>,
    col_h: usize,
    y: &Matrix<i64>,
    col_y: usize,
    limit: usize,
) -> Result<i64, DiophantineError> {
    let mut sum = 0i64;
    for l in 0..limit {
        let term = h[l][col_h]
            .checked_mul(y[l][col_y])
            .ok_or(DiophantineError::Overflow("Overflow in dot product"))?;
        sum = sum
            .checked_add(term)
            .ok_or(DiophantineError::Overflow("Overflow in dot product"))?;
    }
    Ok(sum)
}

/// Solves the linear diophantine system AX = B for an integer matrix X.
///
/// Returns an integer matrix X that is a particular solution to the system.
/// Returns an error if no integer solution exists.
pub fn solve_diophantine(
    a: &Matrix<i64>,
    b: &Matrix<i64>,
) -> Result<Matrix<i64>, DiophantineError> {
    if a.is_empty() {
        return if b.is_empty() {
            Ok(vec![])
        } else {
            Err(DiophantineError::NoSolution(
                "No solution for non-empty B and empty A".to_string(),
            ))
        };
    }
    if a.len() != b.len() {
        return Err(DiophantineError::InvalidDimensions(
            "A and B must have the same number of rows.".to_string(),
        ));
    }

    let a_rows = a.len();
    let a_cols = a[0].len();
    let b_cols = if b[0].is_empty() { 0 } else { b[0].len() };

    let a_t = transpose(a);
    let (h, u) = extended_hnf(&a_t).map_err(|_| DiophantineError::Overflow("HNF overflow"))?;

    let mut pivot_row_of_col = vec![None; a_rows];
    for k in 0..a_cols {
        if let Some(i) = (0..a_rows).find(|&col| h[k][col] != 0) {
            pivot_row_of_col[i] = Some(k);
        }
    }

    let mut y = vec![vec![0i64; b_cols]; a_cols];

    for c in 0..b_cols {
        for i in 0..a_rows {
            match pivot_row_of_col[i] {
                Some(k) => {
                    let pivot = h[k][i];
                    let dot = checked_partial_col_dot(&h, i, &y, c, k)?;
                    let rhs = b[i][c].checked_sub(dot).ok_or(DiophantineError::Overflow(
                        "Overflow during back-substitution",
                    ))?;

                    if rhs % pivot != 0 {
                        return Err(DiophantineError::NoSolution(format!("Failed at row {}", i)));
                    }
                    y[k][c] = rhs / pivot;
                }
                None => {
                    // Equation is redundant, check if it's perfectly consistent
                    let dot = checked_partial_col_dot(&h, i, &y, c, a_cols)?;
                    if dot != b[i][c] {
                        return Err(DiophantineError::NoSolution(format!(
                            "Inconsistent at row {}",
                            i
                        )));
                    }
                }
            }
        }
    }

    // X = U^T * Y
    let u_t = transpose(&u);
    checked_matmul(&u_t, &y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::matmul;

    #[test]
    fn integer_det_simple() {
        // Identity 2x2
        let m = vec![vec![1, 0], vec![0, 1]];
        assert_eq!(integer_det(&m).unwrap(), 1);

        // Swapping rows changes sign
        let m = vec![vec![0, 1], vec![1, 0]];
        assert_eq!(integer_det(&m).unwrap(), -1);

        // 1*4 - 2*3 = -2
        let m = vec![vec![1, 2], vec![3, 4]];
        assert_eq!(integer_det(&m).unwrap(), -2);
    }

    #[test]
    fn integer_det_larger() {
        // 3x3 Singular matrix (row 1 + row 2 = row 3)
        let m = vec![vec![1, 2, 3], vec![4, 5, 6], vec![5, 7, 9]];
        assert_eq!(integer_det(&m).unwrap(), 0);

        // Known determinant
        // | 2 -1  0 |
        // | 0  3 -2 |
        // | 1  4  1 |
        // Det = 2(3+8) - (-1)(0+2) + 0 = 22 + 2 = 24
        let m = vec![vec![2, -1, 0], vec![0, 3, -2], vec![1, 4, 1]];
        assert_eq!(integer_det(&m).unwrap(), 24);
    }

    #[test]
    fn integer_inverse_identity() {
        let m = vec![vec![1, 0], vec![0, 1]];
        let inv = integer_inverse(&m).unwrap();
        assert_eq!(inv, m);
    }

    #[test]
    fn integer_inverse_unimodular() {
        // det = 2*3 - 5*1 = 1
        let m = vec![vec![2, 5], vec![1, 3]];
        let inv = integer_inverse(&m).unwrap();

        assert_eq!(inv, vec![vec![3, -5], vec![-1, 2]]);

        // Use the new matmul function to check A * A^-1 == I
        let prod = matmul(&m, &inv).unwrap();
        assert_eq!(prod, vec![vec![1, 0], vec![0, 1]]);
    }

    #[test]
    fn integer_inverse_swap() {
        // det = -1
        let m = vec![vec![0, 1], vec![1, 0]];
        let inv = integer_inverse(&m).unwrap();
        assert_eq!(inv, vec![vec![0, 1], vec![1, 0]]); // Its own inverse
    }

    #[test]
    fn integer_inverse_not_unimodular() {
        // det = 2
        let m = vec![vec![2, 0], vec![0, 1]];
        let res = integer_inverse(&m);
        assert!(res.is_err());
        assert!(res.unwrap_err() == MatrixError::NotUnimodular(2));
    }

    #[test]
    fn integer_inverse_larger() {
        let m = vec![vec![1, 2, 3], vec![0, 1, 4], vec![5, 6, 0]];
        // det(M) = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
        let inv = integer_inverse(&m).unwrap();

        assert_eq!(
            inv,
            vec![vec![-24, 18, 5], vec![20, -15, -4], vec![-5, 4, 1]]
        );
    }

    #[test]
    fn test_left_kernel() {
        // A = [1 2]
        //     [2 4]
        // Left kernel x * A = 0 should be span of [-2, 1]
        let a = vec![vec![1, 2], vec![2, 4]];
        let ker = left_kernel(&a).unwrap();

        assert_eq!(ker.len(), 1);
        assert_eq!(ker[0], vec![-2, 1]); // Note: Could be [2, -1] depending on pivot signs
    }

    #[test]
    fn test_right_kernel() {
        // A = [1 -1  0]
        //     [0  1 -1]
        // Right kernel A * x = 0 should be span of [1, 1, 1]

        let a = vec![vec![1, -1, 0], vec![0, 1, -1]];
        let ker = right_kernel(&a).unwrap();

        assert_eq!(ker.len(), 3);
        assert_eq!(ker[0].len(), 1);

        // Let's verify A * ker = 0
        let res = matmul(&a, &ker).unwrap();

        assert_eq!(res[0][0], 0);
        assert_eq!(res[1][0], 0);

        let kt = transpose(&ker);
        assert!(kt[0] == vec![1, 1, 1] || kt[0] == vec![-1, -1, -1]);
    }

    #[test]
    fn test_trivial_kernel() {
        // Identity matrix has only trivial 0 vector kernel,
        // so basis should be empty.
        let a = vec![vec![1, 0], vec![0, 1]];
        let ker = left_kernel(&a).unwrap();
        assert!(ker.is_empty());
    }

    #[test]
    fn test_solve_diophantine_simple() {
        // 2x + 3y = 8
        // 4x +  y = 6
        // Solution: x=1, y=2
        let a = vec![vec![2, 3], vec![4, 1]];
        let b = vec![vec![8], vec![6]];
        let x = solve_diophantine(&a, &b).unwrap();
        assert_eq!(x, vec![vec![1], vec![2]]);

        // Verify that A*X = B
        assert_eq!(matmul(&a, &x).unwrap(), b);
    }

    #[test]
    fn test_solve_diophantine_inconsistent() {
        // 2x = 1
        let a = vec![vec![2]];
        let b = vec![vec![1]];
        let res = solve_diophantine(&a, &b);
        assert!(res.is_err());
    }
}
