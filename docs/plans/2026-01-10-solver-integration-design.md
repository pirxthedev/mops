# Solver Integration Design

**Date:** 2026-01-10
**Status:** Draft
**Depends on:** [MOPS Initial Design](2026-01-10-mops-initial-design.md)

## Overview

This document specifies the linear solver integration for MOPS. We support two solver backends:

1. **SuiteSparse** - Direct solver for small-medium problems
2. **hypre** - Iterative solver (AMG preconditioned) for large problems

## Solver Trait

The existing trait in `mops-core/src/solver.rs`:

```rust
pub trait Solver: Send + Sync {
    fn solve(
        &self,
        matrix: &CsrMatrix<f64>,
        rhs: &[f64],
    ) -> Result<(Vec<f64>, SolveStats)>;
}
```

## SuiteSparse Integration

### Crate Selection

Use `suitesparse-sys` or `suitesparse-ldl` for bindings. Evaluation criteria:

| Crate | Pros | Cons |
|-------|------|------|
| `suitesparse-sys` | Direct bindings, full control | Low-level, manual memory |
| `suitesparse-ldl` | Safe wrapper for LDL | Limited to LDL only |
| `cholmod-sys` | Just CHOLMOD | Need multiple crates |

**Recommendation:** Use `suitesparse-src` + `cholmod-sys` for Cholesky, add UMFPACK for non-SPD.

### CHOLMOD for SPD Matrices

FEA stiffness matrices are symmetric positive definite (SPD) after constraint application.

```rust
pub struct CholmodSolver {
    common: cholmod_common,
}

impl Solver for CholmodSolver {
    fn solve(&self, matrix: &CsrMatrix<f64>, rhs: &[f64]) -> Result<(Vec<f64>, SolveStats)> {
        // 1. Convert CsrMatrix to cholmod_sparse
        // 2. Analyze sparsity pattern (reusable for same mesh)
        // 3. Factorize: A = L * L^T
        // 4. Solve: L * L^T * x = b
        // 5. Return solution and stats
    }
}
```

### Symbolic Analysis Caching

For repeated solves with same sparsity pattern (e.g., iterative design):

```rust
pub struct CholmodSolver {
    symbolic: Option<CholmodSymbolic>,  // Cached sparsity analysis
}

impl CholmodSolver {
    pub fn analyze(&mut self, matrix: &CsrMatrix<f64>) {
        // Cache symbolic factorization
    }

    pub fn factorize_and_solve(&self, matrix: &CsrMatrix<f64>, rhs: &[f64]) -> Result<Vec<f64>> {
        // Use cached symbolic if available
    }
}
```

### UMFPACK for Non-SPD

If matrix loses positive-definiteness (e.g., poorly constrained), fallback to LU:

```rust
pub struct UmfpackSolver;

impl Solver for UmfpackSolver {
    fn solve(&self, matrix: &CsrMatrix<f64>, rhs: &[f64]) -> Result<(Vec<f64>, SolveStats)> {
        // LU factorization via UMFPACK
    }
}
```

## hypre Integration

### Crate Selection

| Crate | Status |
|-------|--------|
| `hypre-sys` | Low-level bindings |
| `hypre` | Higher-level wrapper (if available) |

**Recommendation:** Use `hypre-sys` with safe Rust wrapper.

### BoomerAMG Solver

Algebraic Multigrid (AMG) as preconditioner with Conjugate Gradient:

```rust
pub struct HypreSolver {
    config: HypreConfig,
}

pub struct HypreConfig {
    pub max_iterations: usize,    // Default: 1000
    pub tolerance: f64,           // Default: 1e-10
    pub amg_coarsening: CoarseningType,
    pub amg_interpolation: InterpType,
    pub print_level: usize,       // 0 = silent
}

impl Solver for HypreSolver {
    fn solve(&self, matrix: &CsrMatrix<f64>, rhs: &[f64]) -> Result<(Vec<f64>, SolveStats)> {
        // 1. Convert to hypre PARCSR format
        // 2. Create BoomerAMG preconditioner
        // 3. Run PCG iteration
        // 4. Return solution with iteration count, residual
    }
}
```

### AMG Parameters

Reasonable defaults for structural FEA:

```rust
impl Default for HypreConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            amg_coarsening: CoarseningType::HMIS,
            amg_interpolation: InterpType::Ext,
            print_level: 0,
        }
    }
}
```

## Solver Auto-Selection

```rust
pub fn select_solver(n_dofs: usize, config: &SolverConfig) -> Box<dyn Solver> {
    match config.solver_type {
        SolverType::Direct => Box::new(CholmodSolver::new()),
        SolverType::Iterative => Box::new(HypreSolver::new(config.hypre.clone())),
        SolverType::Auto => {
            if n_dofs < config.auto_threshold {
                Box::new(CholmodSolver::new())
            } else {
                Box::new(HypreSolver::new(config.hypre.clone()))
            }
        }
    }
}
```

**Default threshold:** 100,000 DOFs (configurable)

## Matrix Format Conversions

### CsrMatrix to CHOLMOD

```rust
fn to_cholmod_sparse(csr: &CsrMatrix<f64>, common: &mut cholmod_common) -> cholmod_sparse {
    // CHOLMOD expects CSC (compressed sparse column)
    // Option 1: Transpose CSR to get CSC
    // Option 2: Store upper triangle only (symmetric)
}
```

### CsrMatrix to hypre PARCSR

```rust
fn to_parcsr(csr: &CsrMatrix<f64>) -> HYPRE_IJMatrix {
    // hypre uses IJ format for assembly, PARCSR internally
    // Sequential case: single process owns all rows
}
```

## Build Configuration

### Cargo Features

```toml
[features]
default = ["suitesparse"]
suitesparse = ["suitesparse-src", "cholmod-sys"]
hypre = ["hypre-sys"]
all-solvers = ["suitesparse", "hypre"]
```

### System Dependencies

SuiteSparse and hypre require:
- BLAS/LAPACK (OpenBLAS or Intel MKL)
- CMake for building hypre

```toml
[build-dependencies]
cmake = "0.1"

[dependencies]
suitesparse-src = { version = "0.1", features = ["static"] }
hypre-sys = { version = "0.1", optional = true }
```

## Module Structure

```
mops-core/src/
├── solver.rs              # Solver trait, SolverConfig (existing)
├── solver/
│   ├── mod.rs             # Re-exports, select_solver()
│   ├── cholmod.rs         # CHOLMOD wrapper
│   ├── umfpack.rs         # UMFPACK wrapper (fallback)
│   └── hypre.rs           # hypre BoomerAMG wrapper
```

## Error Handling

```rust
pub enum SolverError {
    SingularMatrix,
    NonConvergent { iterations: usize, residual: f64 },
    NumericalInstability,
    AllocationFailed,
    InvalidInput(String),
}
```

## Testing Strategy

### Unit Tests

1. **Small system verification** - Compare to known solution
2. **SPD check** - Verify Cholesky succeeds for valid stiffness matrices
3. **Fallback test** - UMFPACK handles singular-ish matrices

### Performance Tests

1. **Scaling test** - Time vs DOF count
2. **Memory test** - Peak memory vs problem size
3. **Threshold calibration** - Find optimal auto-selection boundary

### Integration Tests

1. **Cantilever beam** - Known displacement at tip
2. **Cube under pressure** - Verify hydrostatic state

## Performance Considerations

### Memory

- Direct solvers: O(n^1.5) fill-in for 3D FEA
- Iterative solvers: O(n) memory

### Time Complexity

- Direct: O(n^2) for 3D FEA (Cholesky)
- Iterative: O(n) per iteration, typically O(n log n) total

### Parallelism

- CHOLMOD: Multi-threaded via BLAS
- hypre: Designed for MPI, but works single-process

## Open Questions

1. **MKL vs OpenBLAS:** MKL faster but licensing complexity
2. **GPU acceleration:** CHOLMOD has CUDA support - worth enabling?
3. **Mixed precision:** Use single precision for AMG setup?

## References

- SuiteSparse User Guide: https://github.com/DrTimothyAldenDavis/SuiteSparse
- hypre Reference Manual: https://hypre.readthedocs.io/
- "Direct Methods for Sparse Linear Systems" - Tim Davis (2006)
