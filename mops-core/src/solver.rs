//! Linear system solvers.
//!
//! Provides direct and iterative solvers for the assembled system Ku = f.

use crate::sparse::CsrMatrix;
use crate::error::{Error, Result};

/// Linear solver interface.
pub trait Solver: Send + Sync {
    /// Solve the linear system Ax = b.
    ///
    /// # Arguments
    ///
    /// * `matrix` - System matrix (K)
    /// * `rhs` - Right-hand side vector (f)
    ///
    /// # Returns
    ///
    /// Solution vector (u)
    fn solve(&self, matrix: &CsrMatrix, rhs: &[f64]) -> Result<Vec<f64>>;

    /// Solver name for diagnostics.
    fn name(&self) -> &str;
}

/// Solver selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    /// Direct solver (Cholesky/LU via SuiteSparse).
    Direct,
    /// Iterative solver (Conjugate Gradient with AMG preconditioner).
    Iterative,
    /// Automatically select based on problem size.
    Auto,
}

impl Default for SolverType {
    fn default() -> Self {
        Self::Auto
    }
}

/// Solver configuration.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Solver type to use.
    pub solver_type: SolverType,
    /// Tolerance for iterative solvers.
    pub tolerance: f64,
    /// Maximum iterations for iterative solvers.
    pub max_iterations: usize,
    /// Problem size threshold for auto-selection (direct below, iterative above).
    pub auto_threshold: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            solver_type: SolverType::Auto,
            tolerance: 1e-10,
            max_iterations: 10000,
            auto_threshold: 100_000, // 100k DOFs
        }
    }
}

/// Solution statistics.
#[derive(Debug, Clone)]
pub struct SolveStats {
    /// Solver name used.
    pub solver: String,
    /// Number of iterations (for iterative solvers).
    pub iterations: Option<usize>,
    /// Final residual norm (for iterative solvers).
    pub residual: Option<f64>,
    /// Wall-clock time in seconds.
    pub time_seconds: f64,
}

/// Placeholder direct solver using nalgebra dense factorization.
///
/// This is a temporary implementation for testing. Production use will
/// integrate SuiteSparse for sparse Cholesky factorization.
pub struct DenseLUSolver;

impl DenseLUSolver {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DenseLUSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for DenseLUSolver {
    fn solve(&self, matrix: &CsrMatrix, rhs: &[f64]) -> Result<Vec<f64>> {
        use nalgebra::{DMatrix, DVector};

        let n = matrix.nrows();
        if n == 0 {
            return Ok(vec![]);
        }

        if n != matrix.ncols() {
            return Err(Error::Solver("Matrix must be square".into()));
        }

        if n != rhs.len() {
            return Err(Error::Solver("RHS size mismatch".into()));
        }

        // Convert to dense (only for small problems - this is a placeholder)
        let dense = DMatrix::from(matrix);
        let b = DVector::from_column_slice(rhs);

        // LU decomposition and solve
        let lu = dense.lu();
        let solution = lu
            .solve(&b)
            .ok_or_else(|| Error::SingularMatrix("LU factorization failed".into()))?;

        Ok(solution.as_slice().to_vec())
    }

    fn name(&self) -> &str {
        "Dense LU (placeholder)"
    }
}

/// Select solver based on configuration and problem size.
pub fn select_solver(config: &SolverConfig, n_dofs: usize) -> Box<dyn Solver> {
    match config.solver_type {
        SolverType::Direct => Box::new(DenseLUSolver::new()),
        SolverType::Iterative => {
            // TODO: Implement iterative solver with hypre
            Box::new(DenseLUSolver::new())
        }
        SolverType::Auto => {
            if n_dofs < config.auto_threshold {
                Box::new(DenseLUSolver::new())
            } else {
                // TODO: Use iterative solver for large problems
                Box::new(DenseLUSolver::new())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::TripletMatrix;
    use approx::assert_relative_eq;

    #[test]
    fn test_dense_lu_simple() {
        // Simple 2x2 system: [2 1; 1 3] * [x; y] = [1; 2]
        // Solution: x = 1/5, y = 3/5
        let mut triplet = TripletMatrix::new(2, 2);
        triplet.add(0, 0, 2.0);
        triplet.add(0, 1, 1.0);
        triplet.add(1, 0, 1.0);
        triplet.add(1, 1, 3.0);

        let matrix = triplet.to_csr();
        let rhs = vec![1.0, 2.0];

        let solver = DenseLUSolver::new();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert_relative_eq!(solution[0], 0.2, epsilon = 1e-10);
        assert_relative_eq!(solution[1], 0.6, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_system() {
        let triplet = TripletMatrix::new(0, 0);
        let matrix = triplet.to_csr();
        let rhs: Vec<f64> = vec![];

        let solver = DenseLUSolver::new();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert!(solution.is_empty());
    }
}
