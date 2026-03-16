//! Error types.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Error type for matrix operations.
pub enum DiophantineError {
    /// Matrices have incompatible dimensions for the requested operation.
    InvalidDimensions(String),
    /// The system is inconsistent or yields non-integer (fractional) solutions.
    NoSolution(String),
    /// An intermediate calculation exceeded the capacity of an `i64`.
    Overflow(&'static str),
}

impl fmt::Display for DiophantineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiophantineError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            DiophantineError::NoSolution(msg) => write!(f, "No integer solution: {}", msg),
            DiophantineError::Overflow(msg) => write!(f, "Integer overflow: {}", msg),
        }
    }
}

impl std::error::Error for DiophantineError {}

/// Error type indicating arithmetic overflow.
#[derive(Debug, Clone)]
pub struct OverflowError;

impl fmt::Display for OverflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Overflow in calculation")
    }
}

impl From<OverflowError> for String {
    fn from(e: OverflowError) -> String {
        format!("{}", e)
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

impl From<MatrixError> for String {
    fn from(e: MatrixError) -> String {
        format!("{}", e)
    }
}
