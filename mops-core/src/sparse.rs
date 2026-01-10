//! Sparse matrix operations.
//!
//! Uses CSR (Compressed Sparse Row) format for efficient matrix-vector products
//! and compatibility with direct/iterative solvers.

use nalgebra_sparse::csr::CsrMatrix as NalgebraCsr;
use std::ops::AddAssign;

/// Compressed Sparse Row matrix.
pub type CsrMatrix = NalgebraCsr<f64>;

/// Builder for assembling a sparse matrix from triplets (COO format).
///
/// Accumulates (row, col, value) triplets and converts to CSR when complete.
pub struct TripletMatrix {
    n_rows: usize,
    n_cols: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    values: Vec<f64>,
}

impl TripletMatrix {
    /// Create a new triplet matrix builder.
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            rows: Vec::new(),
            cols: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create with estimated capacity.
    pub fn with_capacity(n_rows: usize, n_cols: usize, nnz_estimate: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            rows: Vec::with_capacity(nnz_estimate),
            cols: Vec::with_capacity(nnz_estimate),
            values: Vec::with_capacity(nnz_estimate),
        }
    }

    /// Add a value at (row, col). Duplicates are summed during conversion.
    pub fn add(&mut self, row: usize, col: usize, value: f64) {
        debug_assert!(row < self.n_rows, "Row index out of bounds");
        debug_assert!(col < self.n_cols, "Column index out of bounds");

        if value.abs() > f64::EPSILON {
            self.rows.push(row);
            self.cols.push(col);
            self.values.push(value);
        }
    }

    /// Add a dense submatrix at the specified DOF indices.
    ///
    /// This is the core operation for finite element assembly.
    pub fn add_submatrix(&mut self, dof_indices: &[usize], submatrix: &nalgebra::DMatrix<f64>) {
        let n = dof_indices.len();
        debug_assert_eq!(submatrix.nrows(), n);
        debug_assert_eq!(submatrix.ncols(), n);

        for i in 0..n {
            for j in 0..n {
                let value = submatrix[(i, j)];
                if value.abs() > f64::EPSILON {
                    self.rows.push(dof_indices[i]);
                    self.cols.push(dof_indices[j]);
                    self.values.push(value);
                }
            }
        }
    }

    /// Number of stored triplets.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Convert to CSR format, summing duplicate entries.
    pub fn to_csr(self) -> CsrMatrix {
        use nalgebra_sparse::coo::CooMatrix;

        // Build COO matrix first
        let coo = CooMatrix::try_from_triplets(
            self.n_rows,
            self.n_cols,
            self.rows,
            self.cols,
            self.values,
        )
        .expect("Invalid triplet data");

        // Convert to CSR (duplicates are summed)
        CsrMatrix::from(&coo)
    }
}

/// Sparse vector for RHS assembly.
pub struct SparseVector {
    values: Vec<f64>,
}

impl SparseVector {
    /// Create a zero vector of given size.
    pub fn zeros(size: usize) -> Self {
        Self {
            values: vec![0.0; size],
        }
    }

    /// Add a value at the given index.
    pub fn add(&mut self, index: usize, value: f64) {
        self.values[index] += value;
    }

    /// Add values at multiple indices (for element load assembly).
    pub fn add_subvector(&mut self, indices: &[usize], values: &[f64]) {
        debug_assert_eq!(indices.len(), values.len());
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            self.values[idx] += val;
        }
    }

    /// Get the underlying dense vector.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Consume and return the dense vector.
    pub fn into_vec(self) -> Vec<f64> {
        self.values
    }

    /// Mutable access to the underlying vector.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.values
    }
}

impl AddAssign<&SparseVector> for SparseVector {
    fn add_assign(&mut self, rhs: &SparseVector) {
        debug_assert_eq!(self.values.len(), rhs.values.len());
        for (a, b) in self.values.iter_mut().zip(rhs.values.iter()) {
            *a += *b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_triplet_to_csr() {
        let mut triplet = TripletMatrix::new(3, 3);
        triplet.add(0, 0, 1.0);
        triplet.add(1, 1, 2.0);
        triplet.add(2, 2, 3.0);
        triplet.add(0, 1, 0.5);
        triplet.add(1, 0, 0.5);

        let csr = triplet.to_csr();
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.nnz(), 5);
    }

    #[test]
    fn test_duplicate_summation() {
        let mut triplet = TripletMatrix::new(2, 2);
        triplet.add(0, 0, 1.0);
        triplet.add(0, 0, 2.0); // Duplicate - should sum
        triplet.add(0, 0, 3.0); // Another duplicate

        let csr = triplet.to_csr();
        // Check that duplicates were summed
        // The value at (0,0) should be 6.0
        let dense = nalgebra::DMatrix::from(&csr);
        assert!((dense[(0, 0)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_submatrix_assembly() {
        let mut triplet = TripletMatrix::new(6, 6);

        // Simulate element assembly: 2x2 element with dofs [0, 1] and [3, 4]
        let dofs = vec![0, 1, 3, 4];
        let ke = DMatrix::from_row_slice(4, 4, &[
            1.0, 0.5, 0.1, 0.0,
            0.5, 2.0, 0.0, 0.2,
            0.1, 0.0, 1.5, 0.3,
            0.0, 0.2, 0.3, 2.5,
        ]);

        triplet.add_submatrix(&dofs, &ke);

        let csr = triplet.to_csr();
        let dense = nalgebra::DMatrix::from(&csr);

        // Check assembly correctness
        assert!((dense[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((dense[(0, 3)] - 0.1).abs() < 1e-10);
        assert!((dense[(3, 4)] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_vector() {
        let mut vec = SparseVector::zeros(5);
        vec.add(0, 1.0);
        vec.add(2, 3.0);
        vec.add_subvector(&[1, 3], &[2.0, 4.0]);

        let result = vec.as_slice();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 4.0).abs() < 1e-10);
        assert!((result[4] - 0.0).abs() < 1e-10);
    }
}
