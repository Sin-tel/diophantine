//! Error types.

use std::fmt;

/// Error type indicating arithmetic overflow.
#[derive(Debug, Clone)]
pub struct OverflowError;

impl fmt::Display for OverflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Overflow in calculation")
    }
}

/// Error type for matrix operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixError {
    /// Overflow in calculation.
    Overflow,
    /// Singular matrix.
    Singular,
    /// Empty matrix.
    Empty,
    /// Matrix is not square.
    NotSquare,
    /// Matrix is not unimodular.
    NotUnimodular(i64),
    /// Dimension mismatch in matrix multiplication.
    Dimension((usize, usize), (usize, usize)),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            MatrixError::Overflow => write!(f, "Overflow in calculation"),
            MatrixError::Singular => write!(f, "Matrix is singular"),
            MatrixError::Empty => write!(f, "Matrix is empty"),
            MatrixError::NotUnimodular(d) => write!(f, "Matrix is not unimodular (det = {})", d),
            MatrixError::NotSquare => write!(f, "Matrix is not square"),
            MatrixError::Dimension((a, b), (c, d)) => {
                write!(
                    f,
                    "Dimension mismatch: cannot multiply {}x{} by {}x{}",
                    a, b, c, d
                )
            }
        }
    }
}
