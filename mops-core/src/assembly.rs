//! Parallel finite element assembly.
//!
//! Assembles the global stiffness matrix and load vector from element contributions
//! using Rayon for shared-memory parallelism.

use crate::element::Element;
use crate::material::Material;
use crate::mesh::Mesh;
use crate::sparse::{CsrMatrix, SparseVector, TripletMatrix};
use crate::error::Result;
use rayon::prelude::*;
use std::collections::HashMap;

/// Boundary condition types.
#[derive(Debug, Clone)]
pub enum BoundaryCondition {
    /// Fixed displacement (Dirichlet).
    Displacement { node: usize, dof: usize, value: f64 },
    /// Applied force (Neumann).
    Force { node: usize, dof: usize, value: f64 },
}

/// Assembled system ready for solving.
pub struct AssembledSystem {
    /// Global stiffness matrix.
    pub stiffness: CsrMatrix,
    /// Right-hand side (load) vector.
    pub rhs: Vec<f64>,
    /// Number of DOFs in the system.
    pub n_dofs: usize,
    /// Constrained DOF indices and their prescribed values.
    pub constraints: HashMap<usize, f64>,
}

/// Assembly options.
#[derive(Debug, Clone, Default)]
pub struct AssemblyOptions {
    /// Number of parallel threads (0 = auto-detect).
    pub n_threads: usize,
}

/// Assemble global stiffness matrix and load vector.
///
/// This is the main entry point for FEA assembly. It:
/// 1. Computes element stiffness matrices in parallel
/// 2. Assembles into global sparse matrix
/// 3. Applies boundary conditions
///
/// # Arguments
///
/// * `mesh` - Finite element mesh
/// * `elements` - Element implementations indexed by element type
/// * `materials` - Material indexed by element
/// * `boundary_conditions` - Applied BCs (displacements and forces)
/// * `options` - Assembly configuration
pub fn assemble(
    mesh: &Mesh,
    _elements: &[Box<dyn Element>],
    _materials: &[Material],
    boundary_conditions: &[BoundaryCondition],
    _options: &AssemblyOptions,
) -> Result<AssembledSystem> {
    // For now, assume 3 DOFs per node (3D solid elements)
    let dofs_per_node = 3;
    let n_dofs = mesh.n_nodes() * dofs_per_node;

    // Initialize assembly structures
    let triplet = TripletMatrix::with_capacity(n_dofs, n_dofs, n_dofs * 27); // Estimate
    let mut rhs = SparseVector::zeros(n_dofs);
    let mut constraints = HashMap::new();

    // TODO: Parallel element stiffness computation
    // For each element:
    // 1. Get element coordinates from mesh
    // 2. Get element implementation
    // 3. Compute element stiffness matrix
    // 4. Get DOF mapping (element nodes -> global DOFs)
    // 5. Add to triplet matrix

    // Apply boundary conditions
    for bc in boundary_conditions {
        match bc {
            BoundaryCondition::Displacement { node, dof, value } => {
                let global_dof = node * dofs_per_node + dof;
                constraints.insert(global_dof, *value);
            }
            BoundaryCondition::Force { node, dof, value } => {
                let global_dof = node * dofs_per_node + dof;
                rhs.add(global_dof, *value);
            }
        }
    }

    // Convert to CSR
    let stiffness = triplet.to_csr();

    Ok(AssembledSystem {
        stiffness,
        rhs: rhs.into_vec(),
        n_dofs,
        constraints,
    })
}

/// Parallel element stiffness computation.
///
/// Returns (DOF indices, stiffness matrix) pairs for each element.
#[allow(dead_code)]
fn compute_element_stiffnesses(
    mesh: &Mesh,
    elements: &[Box<dyn Element>],
    materials: &[Material],
) -> Vec<(Vec<usize>, nalgebra::DMatrix<f64>)> {
    let dofs_per_node = 3;

    mesh.elements()
        .par_iter()
        .enumerate()
        .map(|(elem_idx, connectivity)| {
            // Get element coordinates
            let coords = mesh.element_coords(elem_idx).unwrap();

            // Get element implementation (TODO: proper element registry)
            let element = &elements[0]; // Placeholder

            // Get material for this element
            let material = &materials[0]; // Placeholder

            // Compute stiffness
            let ke = element.stiffness(&coords, material);

            // Build DOF index map
            let dof_indices: Vec<usize> = connectivity
                .nodes
                .iter()
                .flat_map(|&node| (0..dofs_per_node).map(move |d| node * dofs_per_node + d))
                .collect();

            (dof_indices, ke)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assembly_empty_mesh() {
        let mesh = Mesh::new();
        let elements: Vec<Box<dyn Element>> = vec![];
        let materials: Vec<Material> = vec![];
        let bcs: Vec<BoundaryCondition> = vec![];
        let options = AssemblyOptions::default();

        let result = assemble(&mesh, &elements, &materials, &bcs, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_boundary_condition_application() {
        use nalgebra::Vector3;

        let mut mesh = Mesh::new();
        // Add nodes so we can apply BCs to them
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));

        let elements: Vec<Box<dyn Element>> = vec![];
        let materials: Vec<Material> = vec![];
        let bcs = vec![
            BoundaryCondition::Displacement {
                node: 0,
                dof: 0,
                value: 0.0,
            },
            BoundaryCondition::Force {
                node: 1,
                dof: 2,
                value: 1000.0,
            },
        ];
        let options = AssemblyOptions::default();

        let result = assemble(&mesh, &elements, &materials, &bcs, &options).unwrap();

        // Check constraint was recorded
        assert!(result.constraints.contains_key(&0)); // node 0, dof 0 = global 0

        // Check force was applied (node 1, dof 2 = global DOF 5)
        assert!((result.rhs[5] - 1000.0).abs() < 1e-10);
    }
}
