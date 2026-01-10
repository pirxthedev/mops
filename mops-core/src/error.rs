//! Error types for MOPS operations.

use thiserror::Error;

/// Result type alias using MOPS Error.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during MOPS operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Element-related errors.
    #[error("element error: {0}")]
    Element(String),

    /// Mesh-related errors.
    #[error("mesh error: {0}")]
    Mesh(String),

    /// Assembly errors.
    #[error("assembly error: {0}")]
    Assembly(String),

    /// Solver errors.
    #[error("solver error: {0}")]
    Solver(String),

    /// Matrix singularity or conditioning issues.
    #[error("singular matrix: {0}")]
    SingularMatrix(String),

    /// Invalid material properties.
    #[error("invalid material: {0}")]
    InvalidMaterial(String),

    /// I/O errors (HDF5, file operations).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
