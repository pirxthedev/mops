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
//! - [`IterativeSolver`]: PCG with AMG preconditioning via the kryst library (requires
//!   the `iterative` feature). Best for large problems (>100k DOFs) where direct
//!   factorization becomes memory-prohibitive.
//!
//! # Performance
//!
//! The faer-based sparse Cholesky solver is the production choice for direct solves.
//! For very large problems (>100k DOFs), enable the `iterative` feature to use
//! PCG with AMG preconditioning, which has O(n) memory complexity vs O(n^1.5)
//! for direct methods in 3D FEA.

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;
use faer::linalg::cholesky::llt::factor::LltError;
use faer::prelude::*;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::linalg::LltError as SparseLltError;
use faer::sparse::{SparseColMat, SymbolicSparseColMat};

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

    /// Solve the linear system Ax = b with statistics.
    ///
    /// This is the preferred method when performance metadata is needed.
    /// Default implementation calls `solve()` and returns minimal stats.
    ///
    /// # Arguments
    ///
    /// * `matrix` - System matrix (K)
    /// * `rhs` - Right-hand side vector (f)
    ///
    /// # Returns
    ///
    /// Tuple of (solution vector, solver statistics)
    fn solve_with_stats(&self, matrix: &CsrMatrix, rhs: &[f64]) -> Result<(Vec<f64>, SolveStats)> {
        let start = std::time::Instant::now();
        let solution = self.solve(matrix, rhs)?;
        let elapsed = start.elapsed().as_secs_f64();
        Ok((
            solution,
            SolveStats {
                solver: self.name().to_string(),
                total_time_seconds: elapsed,
                solve_time_seconds: elapsed,
                n_dofs: matrix.nrows(),
                n_nonzeros: matrix.nnz(),
                ..Default::default()
            },
        ))
    }

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

/// Solution statistics including timing and performance metadata.
///
/// This struct is returned alongside the solution vector to provide
/// detailed information about the solve process for logging, HDF5 storage,
/// and performance analysis.
#[derive(Debug, Clone)]
pub struct SolveStats {
    /// Solver name used.
    pub solver: String,
    /// Number of iterations (for iterative solvers, None for direct).
    pub iterations: Option<usize>,
    /// Final residual norm (for iterative solvers, None for direct).
    pub residual: Option<f64>,
    /// Total wall-clock time in seconds (setup + factorization + solve).
    pub total_time_seconds: f64,
    /// Time spent on symbolic analysis/setup (seconds).
    pub setup_time_seconds: f64,
    /// Time spent on numerical factorization (seconds, for direct solvers).
    pub factorization_time_seconds: f64,
    /// Time spent on back-substitution/solve phase (seconds).
    pub solve_time_seconds: f64,
    /// Number of DOFs in the system.
    pub n_dofs: usize,
    /// Number of non-zeros in the matrix.
    pub n_nonzeros: usize,
}

impl SolveStats {
    /// Create a new SolveStats for a direct solver.
    pub fn direct(
        solver: impl Into<String>,
        n_dofs: usize,
        n_nonzeros: usize,
        setup_time: f64,
        factorization_time: f64,
        solve_time: f64,
    ) -> Self {
        Self {
            solver: solver.into(),
            iterations: None,
            residual: None,
            total_time_seconds: setup_time + factorization_time + solve_time,
            setup_time_seconds: setup_time,
            factorization_time_seconds: factorization_time,
            solve_time_seconds: solve_time,
            n_dofs,
            n_nonzeros,
        }
    }

    /// Create a new SolveStats for an iterative solver.
    pub fn iterative(
        solver: impl Into<String>,
        n_dofs: usize,
        n_nonzeros: usize,
        setup_time: f64,
        solve_time: f64,
        iterations: usize,
        residual: f64,
    ) -> Self {
        Self {
            solver: solver.into(),
            iterations: Some(iterations),
            residual: Some(residual),
            total_time_seconds: setup_time + solve_time,
            setup_time_seconds: setup_time,
            factorization_time_seconds: 0.0,
            solve_time_seconds: solve_time,
            n_dofs,
            n_nonzeros,
        }
    }
}

impl Default for SolveStats {
    fn default() -> Self {
        Self {
            solver: String::new(),
            iterations: None,
            residual: None,
            total_time_seconds: 0.0,
            setup_time_seconds: 0.0,
            factorization_time_seconds: 0.0,
            solve_time_seconds: 0.0,
            n_dofs: 0,
            n_nonzeros: 0,
        }
    }
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
        self.solve_with_stats(matrix, rhs)
            .map(|(solution, _)| solution)
    }

    fn solve_with_stats(&self, matrix: &CsrMatrix, rhs: &[f64]) -> Result<(Vec<f64>, SolveStats)> {
        use std::time::Instant;

        let n = matrix.nrows();
        let nnz = matrix.nnz();

        if n == 0 {
            return Ok((vec![], SolveStats::direct(self.name(), 0, 0, 0.0, 0.0, 0.0)));
        }

        if n != matrix.ncols() {
            return Err(Error::Solver("Matrix must be square".into()));
        }

        if n != rhs.len() {
            return Err(Error::Solver("RHS size mismatch".into()));
        }

        // Phase 1: Convert to faer sparse column format (setup)
        let setup_start = Instant::now();
        let csc = csr_to_faer_csc(matrix);
        let csc_ref = csc.as_ref();

        // Perform symbolic analysis
        let symbolic = SymbolicLlt::try_new(csc_ref.symbolic(), faer::Side::Lower)
            .map_err(|_| Error::Solver("Symbolic Cholesky analysis failed".into()))?;
        let setup_time = setup_start.elapsed().as_secs_f64();

        // Phase 2: Numerical factorization
        let factor_start = Instant::now();
        let llt = Llt::try_new_with_symbolic(symbolic, csc_ref, faer::Side::Lower).map_err(
            |e| match e {
                SparseLltError::Generic(err) => {
                    Error::Solver(format!("Sparse Cholesky error: {:?}", err))
                }
                SparseLltError::Numeric(LltError::NonPositivePivot { index }) => {
                    Error::SingularMatrix(format!(
                        "Matrix is not positive definite at pivot {}",
                        index
                    ))
                }
            },
        )?;
        let factor_time = factor_start.elapsed().as_secs_f64();

        // Phase 3: Solve (forward/back substitution)
        let solve_start = Instant::now();
        let mut x = faer::Mat::from_fn(n, 1, |i, _| rhs[i]);
        llt.solve_in_place(x.as_mut());
        let solve_time = solve_start.elapsed().as_secs_f64();

        // Extract solution
        let solution: Vec<f64> = (0..n).map(|i| x[(i, 0)]).collect();

        let stats = SolveStats::direct(self.name(), n, nnz, setup_time, factor_time, solve_time);

        Ok((solution, stats))
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
        let symbolic = self.symbolic.as_ref().ok_or_else(|| {
            Error::Solver("No cached symbolic factorization - call analyze() first".into())
        })?;

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
                SparseLltError::Generic(err) => {
                    Error::Solver(format!("Sparse Cholesky error: {:?}", err))
                }
                SparseLltError::Numeric(LltError::NonPositivePivot { index }) => {
                    Error::SingularMatrix(format!(
                        "Matrix is not positive definite at pivot {}",
                        index
                    ))
                }
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

// ============================================================================
// Iterative solver with AMG preconditioning (requires `iterative` feature)
// ============================================================================

#[cfg(feature = "iterative")]
mod iterative {
    use super::*;
    use kryst::matrix::op::CsrOp;
    use kryst::matrix::CsrMatrix as KrystCsr;
    use kryst::prelude::{KspContext, PcType, SolverType as KrystSolverType};
    use std::sync::Arc;

    /// Configuration for the iterative solver.
    #[derive(Debug, Clone)]
    pub struct IterativeConfig {
        /// Relative tolerance for convergence.
        pub rtol: f64,
        /// Absolute tolerance for convergence.
        pub atol: f64,
        /// Maximum number of iterations.
        pub max_iterations: usize,
        /// Preconditioner type.
        pub preconditioner: PreconditionerType,
    }

    /// Available preconditioner types.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PreconditionerType {
        /// Algebraic Multigrid (best for FEA problems).
        Amg,
        /// Incomplete LU factorization with zero fill-in.
        Ilu0,
        /// Jacobi (diagonal) preconditioner.
        Jacobi,
        /// No preconditioning.
        None,
    }

    impl Default for IterativeConfig {
        fn default() -> Self {
            Self {
                rtol: 1e-10,
                atol: 1e-14,
                max_iterations: 10000,
                preconditioner: PreconditionerType::Amg,
            }
        }
    }

    impl PreconditionerType {
        fn to_kryst(self) -> PcType {
            match self {
                PreconditionerType::Amg => PcType::Amg,
                PreconditionerType::Ilu0 => PcType::Ilu0,
                PreconditionerType::Jacobi => PcType::Jacobi,
                PreconditionerType::None => PcType::None,
            }
        }
    }

    /// Convert mops-core CsrMatrix to kryst CsrMatrix.
    ///
    /// kryst requires column indices to be sorted within each row, which is
    /// guaranteed by our triplet-to-CSR conversion.
    fn to_kryst_csr(matrix: &CsrMatrix) -> KrystCsr<f64> {
        KrystCsr::from_csr(
            matrix.nrows(),
            matrix.ncols(),
            matrix.row_offsets().to_vec(),
            matrix.col_indices().to_vec(),
            matrix.values().to_vec(),
        )
    }

    /// Iterative solver using Preconditioned Conjugate Gradient (PCG) with AMG.
    ///
    /// This solver is optimal for large FEA problems where direct methods become
    /// memory-prohibitive. It uses the kryst library's implementation of PCG with
    /// Algebraic Multigrid (AMG) preconditioning.
    ///
    /// # Memory Complexity
    ///
    /// - O(n) vs O(n^1.5) for direct methods in 3D FEA
    ///
    /// # When to Use
    ///
    /// - Problem size > 100k DOFs
    /// - Memory-constrained environments
    /// - When factorization time dominates (iterative has better startup)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = IterativeConfig::default();
    /// let solver = IterativeSolver::new(config);
    /// let solution = solver.solve(&stiffness_matrix, &force_vector)?;
    /// ```
    pub struct IterativeSolver {
        config: IterativeConfig,
    }

    impl IterativeSolver {
        /// Create a new iterative solver with the given configuration.
        pub fn new(config: IterativeConfig) -> Self {
            Self { config }
        }

        /// Create an iterative solver with default AMG-preconditioned PCG settings.
        pub fn with_amg() -> Self {
            Self::new(IterativeConfig::default())
        }

        /// Create an iterative solver with ILU(0) preconditioning.
        pub fn with_ilu0() -> Self {
            Self::new(IterativeConfig {
                preconditioner: PreconditionerType::Ilu0,
                ..Default::default()
            })
        }
    }

    impl Default for IterativeSolver {
        fn default() -> Self {
            Self::with_amg()
        }
    }

    impl Solver for IterativeSolver {
        fn solve(&self, matrix: &CsrMatrix, rhs: &[f64]) -> Result<Vec<f64>> {
            self.solve_with_stats(matrix, rhs)
                .map(|(solution, _)| solution)
        }

        fn solve_with_stats(
            &self,
            matrix: &CsrMatrix,
            rhs: &[f64],
        ) -> Result<(Vec<f64>, SolveStats)> {
            use std::time::Instant;

            let n = matrix.nrows();
            let nnz = matrix.nnz();

            if n == 0 {
                return Ok((
                    vec![],
                    SolveStats::iterative(self.name(), 0, 0, 0.0, 0.0, 0, 0.0),
                ));
            }

            if n != matrix.ncols() {
                return Err(Error::Solver("Matrix must be square".into()));
            }

            if n != rhs.len() {
                return Err(Error::Solver("RHS size mismatch".into()));
            }

            // Phase 1: Setup (conversion, preconditioner construction)
            let setup_start = Instant::now();

            // Convert to kryst CSR format
            let kryst_csr = to_kryst_csr(matrix);
            let operator = Arc::new(CsrOp::new(Arc::new(kryst_csr)));

            // Set up KSP context
            let mut ksp = KspContext::new();

            // Configure solver type (CG for SPD matrices)
            ksp.set_type(KrystSolverType::Cg)
                .map_err(|e| Error::Solver(format!("Failed to set solver type: {}", e)))?;

            // Configure preconditioner
            ksp.set_pc_type(self.config.preconditioner.to_kryst(), None)
                .map_err(|e| Error::Solver(format!("Failed to set preconditioner: {}", e)))?;

            // Set operator
            ksp.set_operators(operator, None);

            // Configure tolerances
            ksp.rtol = self.config.rtol;
            ksp.atol = self.config.atol;
            ksp.maxits = self.config.max_iterations;

            // Setup (preconditioner factorization)
            ksp.setup()
                .map_err(|e| Error::Solver(format!("Solver setup failed: {}", e)))?;
            let setup_time = setup_start.elapsed().as_secs_f64();

            // Phase 2: Solve (iterative)
            let solve_start = Instant::now();
            let mut solution = vec![0.0; n];
            let ksp_stats = ksp
                .solve(rhs, &mut solution)
                .map_err(|e| Error::Solver(format!("Iterative solve failed: {}", e)))?;
            let solve_time = solve_start.elapsed().as_secs_f64();

            // Extract iteration count and residual from kryst stats
            let iterations = ksp_stats.iterations;
            let residual = ksp_stats.residual_norm;

            let stats = SolveStats::iterative(
                self.name(),
                n,
                nnz,
                setup_time,
                solve_time,
                iterations,
                residual,
            );

            Ok((solution, stats))
        }

        fn name(&self) -> &str {
            match self.config.preconditioner {
                PreconditionerType::Amg => "PCG with AMG (kryst)",
                PreconditionerType::Ilu0 => "PCG with ILU(0) (kryst)",
                PreconditionerType::Jacobi => "PCG with Jacobi (kryst)",
                PreconditionerType::None => "CG unpreconditioned (kryst)",
            }
        }
    }
}

#[cfg(feature = "iterative")]
pub use iterative::{IterativeConfig, IterativeSolver, PreconditionerType};

/// Select solver based on configuration and problem size.
///
/// # Solver Selection Logic
///
/// - `SolverType::Direct`: Always uses sparse Cholesky (faer)
/// - `SolverType::Iterative`: Uses PCG+AMG if the `iterative` feature is enabled,
///   otherwise falls back to direct solver
/// - `SolverType::Auto`: Uses direct for small problems (< `auto_threshold` DOFs),
///   iterative for large problems (if `iterative` feature enabled)
///
/// # Default Threshold
///
/// The default auto-selection threshold is 100,000 DOFs. Below this, direct
/// methods are typically faster. Above this, iterative methods have better
/// memory scaling.
pub fn select_solver(config: &SolverConfig, n_dofs: usize) -> Box<dyn Solver> {
    match config.solver_type {
        SolverType::Direct => Box::new(FaerCholeskySolver::new()),
        SolverType::Iterative => {
            #[cfg(feature = "iterative")]
            {
                Box::new(IterativeSolver::with_amg())
            }
            #[cfg(not(feature = "iterative"))]
            {
                // Fall back to direct solver if iterative feature not enabled
                Box::new(FaerCholeskySolver::new())
            }
        }
        SolverType::Auto => {
            if n_dofs < config.auto_threshold {
                Box::new(FaerCholeskySolver::new())
            } else {
                #[cfg(feature = "iterative")]
                {
                    Box::new(IterativeSolver::with_amg())
                }
                #[cfg(not(feature = "iterative"))]
                {
                    // Fall back to direct solver if iterative feature not enabled
                    Box::new(FaerCholeskySolver::new())
                }
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

// ============================================================================
// Tests for iterative solver (require `iterative` feature)
// ============================================================================

#[cfg(all(test, feature = "iterative"))]
mod iterative_tests {
    use super::*;
    use crate::sparse::TripletMatrix;
    use approx::assert_relative_eq;

    #[test]
    fn test_iterative_solver_simple_spd() {
        // Simple 2x2 SPD system: [4 2; 2 3] * [x; y] = [4; 5]
        // Solution: x = 0.25, y = 1.5
        let mut triplet = TripletMatrix::new(2, 2);
        triplet.add(0, 0, 4.0);
        triplet.add(0, 1, 2.0);
        triplet.add(1, 0, 2.0);
        triplet.add(1, 1, 3.0);

        let matrix = triplet.to_csr();
        let rhs = vec![4.0, 5.0];

        let solver = IterativeSolver::with_amg();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert_relative_eq!(solution[0], 0.25, epsilon = 1e-8);
        assert_relative_eq!(solution[1], 1.5, epsilon = 1e-8);
    }

    #[test]
    fn test_iterative_solver_jacobi() {
        // Test with Jacobi preconditioner
        let mut triplet = TripletMatrix::new(2, 2);
        triplet.add(0, 0, 4.0);
        triplet.add(0, 1, 2.0);
        triplet.add(1, 0, 2.0);
        triplet.add(1, 1, 3.0);

        let matrix = triplet.to_csr();
        let rhs = vec![4.0, 5.0];

        let config = IterativeConfig {
            preconditioner: PreconditionerType::Jacobi,
            ..Default::default()
        };
        let solver = IterativeSolver::new(config);
        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert_relative_eq!(solution[0], 0.25, epsilon = 1e-8);
        assert_relative_eq!(solution[1], 1.5, epsilon = 1e-8);
    }

    #[test]
    fn test_iterative_solver_identity() {
        // Identity matrix: solution equals RHS
        let mut triplet = TripletMatrix::new(4, 4);
        for i in 0..4 {
            triplet.add(i, i, 1.0);
        }

        let matrix = triplet.to_csr();
        let rhs = vec![1.0, 2.0, 3.0, 4.0];

        let solver = IterativeSolver::with_amg();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        for i in 0..4 {
            assert_relative_eq!(solution[i], rhs[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_iterative_solver_diagonal() {
        // Diagonal SPD matrix
        let mut triplet = TripletMatrix::new(3, 3);
        triplet.add(0, 0, 2.0);
        triplet.add(1, 1, 3.0);
        triplet.add(2, 2, 4.0);

        let matrix = triplet.to_csr();
        let rhs = vec![6.0, 9.0, 8.0];

        let solver = IterativeSolver::with_amg();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert_relative_eq!(solution[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(solution[1], 3.0, epsilon = 1e-8);
        assert_relative_eq!(solution[2], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_iterative_solver_empty() {
        let triplet = TripletMatrix::new(0, 0);
        let matrix = triplet.to_csr();
        let rhs: Vec<f64> = vec![];

        let solver = IterativeSolver::with_amg();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert!(solution.is_empty());
    }

    #[test]
    fn test_iterative_solver_rhs_mismatch() {
        let mut triplet = TripletMatrix::new(2, 2);
        triplet.add(0, 0, 1.0);
        triplet.add(1, 1, 1.0);

        let matrix = triplet.to_csr();
        let rhs = vec![1.0, 2.0, 3.0]; // Wrong size

        let solver = IterativeSolver::with_amg();
        assert!(solver.solve(&matrix, &rhs).is_err());
    }

    #[test]
    fn test_iterative_solver_fea_like_matrix() {
        // A small FEA-like stiffness matrix (tridiagonal banded structure)
        let n = 20;
        let mut triplet = TripletMatrix::new(n, n);

        // Diagonal dominance (required for SPD)
        for i in 0..n {
            triplet.add(i, i, 4.0);
        }

        // Off-diagonal coupling (symmetric)
        for i in 0..(n - 1) {
            triplet.add(i, i + 1, -1.0);
            triplet.add(i + 1, i, -1.0);
        }

        let matrix = triplet.to_csr();
        let rhs: Vec<f64> = (0..n)
            .map(|i| if i == 0 || i == n - 1 { 1.0 } else { 0.0 })
            .collect();

        let solver = IterativeSolver::with_amg();
        let solution = solver.solve(&matrix, &rhs).unwrap();

        // Verify solution by computing residual ||Ax - b||
        assert!(solution.len() == n);
        assert!(solution.iter().all(|&x| x.is_finite()));

        let dense = nalgebra::DMatrix::from(&matrix);
        let x_vec = nalgebra::DVector::from_vec(solution);
        let b_vec = nalgebra::DVector::from_vec(rhs);
        let residual = (&dense * &x_vec - &b_vec).norm();
        // Iterative solver tolerances are looser than direct solvers
        assert!(residual < 1e-4, "Residual too large: {}", residual);
    }

    #[test]
    fn test_select_solver_iterative_for_large() {
        let config = SolverConfig {
            solver_type: SolverType::Auto,
            auto_threshold: 10, // Very low threshold for testing
            ..Default::default()
        };

        // Below threshold: direct solver
        let solver_small = select_solver(&config, 5);
        assert_eq!(solver_small.name(), "faer Sparse Cholesky (LLᵀ)");

        // Above threshold: iterative solver
        let solver_large = select_solver(&config, 100);
        assert_eq!(solver_large.name(), "PCG with AMG (kryst)");
    }

    #[test]
    fn test_select_solver_explicit_iterative() {
        let config = SolverConfig {
            solver_type: SolverType::Iterative,
            ..Default::default()
        };

        let solver = select_solver(&config, 10);
        assert_eq!(solver.name(), "PCG with AMG (kryst)");
    }

    #[test]
    fn test_iterative_solver_name() {
        let amg = IterativeSolver::with_amg();
        assert_eq!(amg.name(), "PCG with AMG (kryst)");

        let ilu = IterativeSolver::with_ilu0();
        assert_eq!(ilu.name(), "PCG with ILU(0) (kryst)");

        let jacobi = IterativeSolver::new(IterativeConfig {
            preconditioner: PreconditionerType::Jacobi,
            ..Default::default()
        });
        assert_eq!(jacobi.name(), "PCG with Jacobi (kryst)");
    }
}
