//! Linear system solvers.
//!
//! Provides direct and iterative solvers for the assembled system Ku = f.
//!
//! # Solver Backends
//!
//! - [`FaerCholeskySolver`]: Sparse Cholesky factorization using the faer library.
//!   Best for symmetric positive definite (SPD) matrices, which stiffness matrices
//!   are after constraint application.
//! - [`DenseLUSolver`]: Placeholder using nalgebra dense LU (for small test problems only).
//!
//! # Performance
//!
//! The faer-based sparse Cholesky solver is the production choice for direct solves.
//! For very large problems (>100k DOFs), an iterative solver with AMG preconditioning
//! would be more efficient, but that's not yet implemented.

use crate::sparse::CsrMatrix;
use crate::error::{Error, Result};
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::linalg::LltError as SparseLltError;
use faer::linalg::cholesky::llt::factor::LltError;
use faer::prelude::*;

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

/// Convert nalgebra-sparse CSR matrix to faer SparseColMat (CSC format).
///
/// FEA stiffness matrices are symmetric, so CSR and CSC transposed are equivalent.
/// faer expects CSC format, which we get by treating CSR as CSC of the transpose.
/// Since K is symmetric, K^T = K, so this works directly.
fn csr_to_faer_csc(csr: &CsrMatrix) -> SparseColMat<usize, f64> {
    let nrows = csr.nrows();
    let ncols = csr.ncols();

    // For symmetric matrices, we can convert CSR to CSC by transposing indices
    // CSR format: row_offsets, col_indices, values (row-major)
    // CSC format: col_offsets, row_indices, values (column-major)
    // For symmetric K, CSR(K) = CSC(K^T) = CSC(K)

    // Extract CSR data
    let row_offsets = csr.row_offsets();
    let col_indices = csr.col_indices();
    let values = csr.values();

    // Build CSC by transposing: each CSR row becomes a CSC column
    // First, count entries per column
    let mut col_counts = vec![0usize; ncols];
    for &col in col_indices {
        col_counts[col] += 1;
    }

    // Compute column offsets
    let mut col_offsets = vec![0usize; ncols + 1];
    for i in 0..ncols {
        col_offsets[i + 1] = col_offsets[i] + col_counts[i];
    }

    // Fill row indices and values for CSC
    let nnz = values.len();
    let mut csc_row_indices = vec![0usize; nnz];
    let mut csc_values = vec![0.0f64; nnz];
    let mut col_positions = col_offsets[..ncols].to_vec();

    for row in 0..nrows {
        let row_start = row_offsets[row];
        let row_end = row_offsets[row + 1];
        for idx in row_start..row_end {
            let col = col_indices[idx];
            let val = values[idx];
            let pos = col_positions[col];
            csc_row_indices[pos] = row;
            csc_values[pos] = val;
            col_positions[col] += 1;
        }
    }

    // Create faer symbolic structure and sparse matrix
    // SAFETY: We've constructed valid CSC data
    unsafe {
        SparseColMat::new(
            SymbolicSparseColMat::new_unchecked(nrows, ncols, col_offsets, None, csc_row_indices),
            csc_values,
        )
    }
}

/// Sparse Cholesky solver using the faer library.
///
/// This is the production sparse direct solver for MOPS. It uses faer's
/// sparse LLᵀ (Cholesky) factorization, which is efficient for the symmetric
/// positive definite matrices that arise from finite element stiffness matrices.
///
/// # Example
///
/// ```ignore
/// let solver = FaerCholeskySolver::new();
/// let solution = solver.solve(&stiffness_matrix, &force_vector)?;
/// ```
pub struct FaerCholeskySolver;

impl FaerCholeskySolver {
    /// Create a new sparse Cholesky solver.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FaerCholeskySolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for FaerCholeskySolver {
    fn solve(&self, matrix: &CsrMatrix, rhs: &[f64]) -> Result<Vec<f64>> {
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

        // Convert to faer sparse column format
        let csc = csr_to_faer_csc(matrix);
        let csc_ref = csc.as_ref();

        // Perform symbolic analysis
        let symbolic = SymbolicLlt::try_new(csc_ref.symbolic(), faer::Side::Lower)
            .map_err(|_| Error::Solver("Symbolic Cholesky analysis failed".into()))?;

        // Numerical factorization
        let llt = Llt::try_new_with_symbolic(symbolic, csc_ref, faer::Side::Lower)
            .map_err(|e| match e {
                SparseLltError::Generic(err) => Error::Solver(
                    format!("Sparse Cholesky error: {:?}", err)
                ),
                SparseLltError::Numeric(LltError::NonPositivePivot { index }) => Error::SingularMatrix(
                    format!("Matrix is not positive definite at pivot {}", index)
                ),
            })?;

        // Solve the system
        let mut x = faer::Mat::from_fn(n, 1, |i, _| rhs[i]);
        llt.solve_in_place(x.as_mut());

        // Extract solution
        Ok((0..n).map(|i| x[(i, 0)]).collect())
    }

    fn name(&self) -> &str {
        "faer Sparse Cholesky (LLᵀ)"
    }
}

/// Sparse Cholesky solver with cached symbolic factorization.
///
/// For repeated solves with the same sparsity pattern (e.g., iterative design
/// or multiple load cases), this caches the symbolic analysis for reuse.
pub struct CachedCholeskySolver {
    symbolic: Option<SymbolicLlt<usize>>,
}

impl CachedCholeskySolver {
    /// Create a new cached Cholesky solver.
    pub fn new() -> Self {
        Self { symbolic: None }
    }

    /// Perform symbolic analysis on the matrix sparsity pattern.
    ///
    /// Call this once, then use `solve_with_cached_symbolic` for repeated solves.
    pub fn analyze(&mut self, matrix: &CsrMatrix) -> Result<()> {
        let csc = csr_to_faer_csc(matrix);
        let symbolic = SymbolicLlt::try_new(csc.as_ref().symbolic(), faer::Side::Lower)
            .map_err(|_| Error::Solver("Symbolic Cholesky analysis failed".into()))?;
        self.symbolic = Some(symbolic);
        Ok(())
    }

    /// Solve using the cached symbolic factorization.
    ///
    /// The matrix must have the same sparsity pattern as the one used for `analyze`.
    pub fn solve_with_cached_symbolic(&self, matrix: &CsrMatrix, rhs: &[f64]) -> Result<Vec<f64>> {
        let symbolic = self.symbolic.as_ref()
            .ok_or_else(|| Error::Solver("No cached symbolic factorization - call analyze() first".into()))?;

        let n = matrix.nrows();
        if n == 0 {
            return Ok(vec![]);
        }

        if n != rhs.len() {
            return Err(Error::Solver("RHS size mismatch".into()));
        }

        let csc = csr_to_faer_csc(matrix);

        // Numerical factorization with cached symbolic
        let llt = Llt::try_new_with_symbolic(symbolic.clone(), csc.as_ref(), faer::Side::Lower)
            .map_err(|e| match e {
                SparseLltError::Generic(err) => Error::Solver(
                    format!("Sparse Cholesky error: {:?}", err)
                ),
                SparseLltError::Numeric(LltError::NonPositivePivot { index }) => Error::SingularMatrix(
                    format!("Matrix is not positive definite at pivot {}", index)
                ),
            })?;

        // Solve
        let mut x = faer::Mat::from_fn(n, 1, |i, _| rhs[i]);
        llt.solve_in_place(x.as_mut());

        Ok((0..n).map(|i| x[(i, 0)]).collect())
    }
}

impl Default for CachedCholeskySolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for CachedCholeskySolver {
    fn solve(&self, matrix: &CsrMatrix, rhs: &[f64]) -> Result<Vec<f64>> {
        // If we have cached symbolic, use it; otherwise do full solve
        if self.symbolic.is_some() {
            self.solve_with_cached_symbolic(matrix, rhs)
        } else {
            // Fall back to non-cached solve
            FaerCholeskySolver::new().solve(matrix, rhs)
        }
    }

    fn name(&self) -> &str {
        "faer Sparse Cholesky (cached)"
    }
}

/// Select solver based on configuration and problem size.
pub fn select_solver(config: &SolverConfig, n_dofs: usize) -> Box<dyn Solver> {
    match config.solver_type {
        SolverType::Direct => Box::new(FaerCholeskySolver::new()),
        SolverType::Iterative => {
            // TODO: Implement iterative solver with hypre
            // For now, fall back to direct solver
            Box::new(FaerCholeskySolver::new())
        }
        SolverType::Auto => {
            if n_dofs < config.auto_threshold {
                Box::new(FaerCholeskySolver::new())
            } else {
                // TODO: Use iterative solver for large problems
                // For now, use direct solver for all sizes
                Box::new(FaerCholeskySolver::new())
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

    #[test]
    fn test_faer_cholesky_simple_spd() {
        // Simple 2x2 SPD system: [4 2; 2 3] * [x; y] = [4; 5]
        // Solution: x = 0.25, y = 1.5
        let mut triplet = TripletMatrix::new(2, 2);
        triplet.add(0, 0, 4.0);
        triplet.add(0, 1, 2.0);
        triplet.add(1, 0, 2.0);
        triplet.add(1, 1, 3.0);

        let matrix = triplet.to_csr();
        let rhs = vec![4.0, 5.0];

        let solver = FaerCholeskySolver::new();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert_relative_eq!(solution[0], 0.25, epsilon = 1e-10);
        assert_relative_eq!(solution[1], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_faer_cholesky_3x3_spd() {
        // 3x3 SPD system (positive definite by construction)
        // A = [4 2 0; 2 5 2; 0 2 3]
        // b = [2; 8; 5]
        // Solution: x = [-3/16, 11/8, 3/4] = [-0.1875, 1.375, 0.75]
        let mut triplet = TripletMatrix::new(3, 3);
        triplet.add(0, 0, 4.0);
        triplet.add(0, 1, 2.0);
        triplet.add(1, 0, 2.0);
        triplet.add(1, 1, 5.0);
        triplet.add(1, 2, 2.0);
        triplet.add(2, 1, 2.0);
        triplet.add(2, 2, 3.0);

        let matrix = triplet.to_csr();
        let rhs = vec![2.0, 8.0, 5.0];

        let solver = FaerCholeskySolver::new();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        // Verify Ax = b
        let expected = [-0.1875, 1.375, 0.75]; // Exact: -3/16, 11/8, 3/4
        for i in 0..3 {
            assert_relative_eq!(solution[i], expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_faer_cholesky_identity() {
        // Identity matrix: solution equals RHS
        let mut triplet = TripletMatrix::new(4, 4);
        for i in 0..4 {
            triplet.add(i, i, 1.0);
        }

        let matrix = triplet.to_csr();
        let rhs = vec![1.0, 2.0, 3.0, 4.0];

        let solver = FaerCholeskySolver::new();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        for i in 0..4 {
            assert_relative_eq!(solution[i], rhs[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_faer_cholesky_diagonal() {
        // Diagonal SPD matrix
        let mut triplet = TripletMatrix::new(3, 3);
        triplet.add(0, 0, 2.0);
        triplet.add(1, 1, 3.0);
        triplet.add(2, 2, 4.0);

        let matrix = triplet.to_csr();
        let rhs = vec![6.0, 9.0, 8.0];

        let solver = FaerCholeskySolver::new();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert_relative_eq!(solution[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(solution[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(solution[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_faer_cholesky_empty() {
        let triplet = TripletMatrix::new(0, 0);
        let matrix = triplet.to_csr();
        let rhs: Vec<f64> = vec![];

        let solver = FaerCholeskySolver::new();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert!(solution.is_empty());
    }

    #[test]
    fn test_faer_cholesky_rhs_mismatch() {
        let mut triplet = TripletMatrix::new(2, 2);
        triplet.add(0, 0, 1.0);
        triplet.add(1, 1, 1.0);

        let matrix = triplet.to_csr();
        let rhs = vec![1.0, 2.0, 3.0]; // Wrong size

        let solver = FaerCholeskySolver::new();
        assert!(solver.solve(&matrix, &rhs).is_err());
    }

    #[test]
    fn test_faer_cholesky_not_positive_definite() {
        // Non-positive definite matrix (negative eigenvalue)
        let mut triplet = TripletMatrix::new(2, 2);
        triplet.add(0, 0, 1.0);
        triplet.add(0, 1, 2.0);
        triplet.add(1, 0, 2.0);
        triplet.add(1, 1, 1.0); // Not SPD: eigenvalues are 3 and -1

        let matrix = triplet.to_csr();
        let rhs = vec![1.0, 1.0];

        let solver = FaerCholeskySolver::new();
        let result = solver.solve(&matrix, &rhs);
        assert!(result.is_err());
    }

    #[test]
    fn test_cached_cholesky_solver() {
        // Test the cached solver
        let mut triplet = TripletMatrix::new(2, 2);
        triplet.add(0, 0, 4.0);
        triplet.add(0, 1, 2.0);
        triplet.add(1, 0, 2.0);
        triplet.add(1, 1, 3.0);

        let matrix = triplet.to_csr();
        let rhs1 = vec![4.0, 5.0];
        let rhs2 = vec![8.0, 10.0]; // 2x the first RHS

        let mut solver = CachedCholeskySolver::new();
        solver.analyze(&matrix).unwrap();

        let solution1 = solver.solve_with_cached_symbolic(&matrix, &rhs1).unwrap();
        let solution2 = solver.solve_with_cached_symbolic(&matrix, &rhs2).unwrap();

        // Solution should scale linearly
        assert_relative_eq!(solution1[0], 0.25, epsilon = 1e-10);
        assert_relative_eq!(solution1[1], 1.5, epsilon = 1e-10);
        assert_relative_eq!(solution2[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(solution2[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_select_solver_uses_faer() {
        let config = SolverConfig::default();
        let solver = select_solver(&config, 100);
        assert_eq!(solver.name(), "faer Sparse Cholesky (LLᵀ)");
    }

    #[test]
    fn test_fea_like_stiffness_matrix() {
        // A small FEA-like stiffness matrix (6x6, representing 2 elements)
        // This simulates a tridiagonal-like banded structure common in 1D FEA
        let mut triplet = TripletMatrix::new(6, 6);

        // Diagonal dominance (required for SPD)
        for i in 0..6 {
            triplet.add(i, i, 4.0);
        }

        // Off-diagonal coupling (symmetric)
        for i in 0..5 {
            triplet.add(i, i + 1, -1.0);
            triplet.add(i + 1, i, -1.0);
        }

        let matrix = triplet.to_csr();
        let rhs = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let solver = FaerCholeskySolver::new();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        // Verify solution by checking Ax ≈ b
        // For this simple system, we verify the solution satisfies the equation
        assert!(solution.len() == 6);
        assert!(solution.iter().all(|&x| x.is_finite()));

        // Compute residual ||Ax - b||
        let dense = nalgebra::DMatrix::from(&matrix);
        let x_vec = nalgebra::DVector::from_vec(solution.clone());
        let b_vec = nalgebra::DVector::from_vec(rhs.clone());
        let residual = (&dense * &x_vec - &b_vec).norm();
        assert!(residual < 1e-10, "Residual too large: {}", residual);
    }
}
