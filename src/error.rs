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

impl From<DiophantineError> for String {
    fn from(e: DiophantineError) -> String {
        format!("{}", e)
    }
}
