use crate::Matrix;
use num_traits::{One, Zero};
use std::fmt::Display;
use std::ops::SubAssign;

pub fn eye<T>(n: usize) -> Matrix<T>
where
    T: One + Zero + Copy,
{
    let mut m = vec![vec![T::zero(); n]; n];
    for i in 0..n {
        m[i][i] = T::one();
    }
    m
}

pub fn pretty_print<T: Display>(matrix: &Matrix<T>) {
    if matrix.is_empty() {
        println!("[[]]");
        return;
    }

    // Convert all values to strings and find the max width per column
    let stringified: Vec<Vec<String>> = matrix
        .iter()
        .map(|row| row.iter().map(|v| v.to_string()).collect())
        .collect();

    let col_count = stringified.iter().map(|r| r.len()).max().unwrap_or(0);

    let col_widths: Vec<usize> = (0..col_count)
        .map(|c| {
            stringified
                .iter()
                .filter_map(|row| row.get(c))
                .map(|s| s.len())
                .max()
                .unwrap_or(0)
        })
        .collect();

    let n = stringified.len();
    for (i, row) in stringified.iter().enumerate() {
        let cells: String = (0..col_count)
            .map(|c| {
                let s = row.get(c).map(String::as_str).unwrap_or("");
                format!(" {:>width$} ", s, width = col_widths[c])
            })
            .collect::<Vec<_>>()
            .join("");

        if i == 0 {
            println!("[[{}]", cells);
        } else if i == n - 1 {
            println!(" [{}]]", cells);
        } else {
            println!(" [{}]", cells);
        }
    }
}

/// Subtracts `scale * other` from `target` in place.
pub(crate) fn vec_sub_assign<T>(target: &mut [T], other: &[T], scale: T)
where
    T: Copy + SubAssign + std::ops::Mul<Output = T>,
{
    for (t, &o) in target.iter_mut().zip(other.iter()) {
        *t -= scale * o;
    }
}

/// Multiplies two integer matrices (A * B).
/// Returns an error if the dimensions do not match or if an integer overflow occurs.
pub fn matmul(a: &Matrix<i64>, b: &Matrix<i64>) -> Result<Matrix<i64>, String> {
    let a_rows = a.len();
    if a_rows == 0 {
        return Ok(vec![]);
    }
    let a_cols = a[0].len();

    let b_rows = b.len();
    if b_rows == 0 {
        return Ok(vec![]);
    }
    let b_cols = b[0].len();

    if a_cols != b_rows {
        return Err(format!(
            "Dimension mismatch: cannot multiply {}x{} by {}x{}",
            a_rows, a_cols, b_rows, b_cols
        ));
    }

    let mut result = vec![vec![0; b_cols]; a_rows];

    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum: i64 = 0;
            for k in 0..a_cols {
                let term = a[i][k]
                    .checked_mul(b[k][j])
                    .ok_or_else(|| "Overflow during matrix multiplication".to_string())?;

                sum = sum
                    .checked_add(term)
                    .ok_or_else(|| "Overflow during matrix multiplication".to_string())?;
            }
            result[i][j] = sum;
        }
    }

    Ok(result)
}

/// Helper to transpose a matrix
pub fn transpose(matrix: &Matrix<i64>) -> Matrix<i64> {
    if matrix.is_empty() {
        return vec![];
    }
    let n = matrix.len();
    let m = matrix[0].len();
    let mut t = vec![vec![0; n]; m];
    for i in 0..n {
        for j in 0..m {
            t[j][i] = matrix[i][j];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_dimensions() {
        let a = vec![vec![1, 2, 3], vec![4, 5, 6]]; // 2x3
        let b = vec![vec![7, 8], vec![9, 10], vec![11, 12]]; // 3x2

        let prod = matmul(&a, &b).unwrap();
        assert_eq!(prod, vec![vec![58, 64], vec![139, 154]]);

        // Invalid dimension check
        assert!(matmul(&b, &b).is_err());
    }
}
