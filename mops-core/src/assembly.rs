//! Parallel finite element assembly.
//!
//! Assembles the global stiffness matrix and load vector from element contributions
//! using Rayon for shared-memory parallelism.

use crate::element::{create_element, create_element_with_thickness};
use crate::error::Result;
use crate::material::Material;
use crate::mesh::Mesh;
use crate::sparse::{CsrMatrix, SparseVector, TripletMatrix};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

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
#[derive(Debug, Clone)]
pub struct AssemblyOptions {
    /// Number of parallel threads (0 = auto-detect).
    pub n_threads: usize,
    /// Thickness for 2D plane stress elements (default: 1.0).
    /// This is used when creating 2D elements (Tri3, Tri6, Quad4, Quad8).
    /// Ignored for 3D elements (Tet4, Tet10, Hex8, Hex20).
    pub thickness: f64,
}

impl Default for AssemblyOptions {
    fn default() -> Self {
        Self {
            n_threads: 0,
            thickness: 1.0,
        }
    }
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
/// * `material` - Material properties (uniform for all elements)
/// * `boundary_conditions` - Applied BCs (displacements and forces)
/// * `options` - Assembly configuration
///
/// # Example
///
/// ```ignore
/// use mops_core::assembly::{assemble, AssemblyOptions, BoundaryCondition};
/// use mops_core::material::Material;
/// use mops_core::mesh::{Mesh, ElementType};
/// use nalgebra::Vector3;
///
/// let mut mesh = Mesh::new();
/// // Add nodes and elements...
///
/// let material = Material::steel();
/// let bcs = vec![
///     BoundaryCondition::Displacement { node: 0, dof: 0, value: 0.0 },
///     BoundaryCondition::Force { node: 3, dof: 2, value: 1000.0 },
/// ];
///
/// let system = assemble(&mesh, &material, &bcs, &AssemblyOptions::default()).unwrap();
/// ```
pub fn assemble(
    mesh: &Mesh,
    material: &Material,
    boundary_conditions: &[BoundaryCondition],
    options: &AssemblyOptions,
) -> Result<AssembledSystem> {
    // Determine DOFs per node from mesh element types
    let dofs_per_node = mesh.dofs_per_node()?;
    let n_dofs = mesh.n_nodes() * dofs_per_node;

    // Estimate non-zeros based on element dimension
    // 3D: ~27 per DOF (3x3x3 stencil)
    // 2D: ~9 per DOF (3x3 stencil)
    let nnz_per_dof = if dofs_per_node == 3 { 27 } else { 9 };
    let nnz_estimate = n_dofs * nnz_per_dof;
    let triplet = Mutex::new(TripletMatrix::with_capacity(n_dofs, n_dofs, nnz_estimate));

    // Get thickness for 2D elements from options
    let thickness = options.thickness;

    // Parallel element stiffness computation and assembly
    mesh.elements()
        .par_iter()
        .enumerate()
        .for_each(|(elem_idx, connectivity)| {
            // Create element implementation from type
            // Use thickness for 2D elements, standard creation for 3D
            let element = if connectivity.element_type.dimension() == 2 {
                create_element_with_thickness(connectivity.element_type, thickness)
            } else {
                create_element(connectivity.element_type)
            };

            // Get element nodal coordinates
            let coords = mesh.element_coords(elem_idx).expect("Valid element index");

            // Compute element stiffness matrix
            let ke = element.stiffness(&coords, material);

            // Build DOF index mapping: element nodes -> global DOFs
            // Use the element's dofs_per_node to ensure correct mapping
            let elem_dofs_per_node = element.dofs_per_node();
            let dof_indices: Vec<usize> = connectivity
                .nodes
                .iter()
                .flat_map(|&node| (0..elem_dofs_per_node).map(move |d| node * dofs_per_node + d))
                .collect();

            // Add to global triplet matrix (thread-safe via mutex)
            triplet.lock().unwrap().add_submatrix(&dof_indices, &ke);
        });

    // Extract assembled triplet matrix
    let triplet = triplet.into_inner().unwrap();

    // Initialize RHS and constraints
    let mut rhs = SparseVector::zeros(n_dofs);
    let mut constraints = HashMap::new();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ElementType;
    use nalgebra::Vector3;

    #[test]
    fn test_assembly_empty_mesh() {
        let mesh = Mesh::new();
        let material = Material::steel();
        let bcs: Vec<BoundaryCondition> = vec![];
        let options = AssemblyOptions::default();

        let result = assemble(&mesh, &material, &bcs, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_boundary_condition_application() {
        let mut mesh = Mesh::new();
        // Add nodes so we can apply BCs to them
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));

        let material = Material::steel();
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

        let result = assemble(&mesh, &material, &bcs, &options).unwrap();

        // Check constraint was recorded
        assert!(result.constraints.contains_key(&0)); // node 0, dof 0 = global 0

        // Check force was applied (node 1, dof 2 = global DOF 5)
        assert!((result.rhs[5] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_tet4_assembly() {
        // Create a single tetrahedron mesh
        let mut mesh = Mesh::new();

        // Unit tetrahedron nodes
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 0.0, 1.0));

        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3])
            .unwrap();

        // Steel-like material
        let material = Material::steel();
        let bcs: Vec<BoundaryCondition> = vec![];
        let options = AssemblyOptions::default();

        let result = assemble(&mesh, &material, &bcs, &options).unwrap();

        // 4 nodes * 3 DOFs = 12 DOFs
        assert_eq!(result.n_dofs, 12);

        // Stiffness matrix should have non-zeros
        assert!(result.stiffness.nnz() > 0);

        // Stiffness should be symmetric (check a few entries)
        let dense = nalgebra::DMatrix::from(&result.stiffness);
        for i in 0..12 {
            for j in 0..12 {
                assert!(
                    (dense[(i, j)] - dense[(j, i)]).abs() < 1e-6,
                    "Stiffness not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Diagonal entries should be positive (SPD matrix)
        for i in 0..12 {
            assert!(dense[(i, i)] > 0.0, "Diagonal {} is not positive", i);
        }
    }

    #[test]
    fn test_single_hex8_assembly() {
        // Create a single hexahedron mesh (unit cube)
        let mut mesh = Mesh::new();

        // Unit cube nodes
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 1.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 0.0, 1.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 1.0));
        mesh.add_node(Vector3::new(1.0, 1.0, 1.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 1.0));

        mesh.add_element(ElementType::Hex8, vec![0, 1, 2, 3, 4, 5, 6, 7])
            .unwrap();

        let material = Material::steel();
        let bcs: Vec<BoundaryCondition> = vec![];
        let options = AssemblyOptions::default();

        let result = assemble(&mesh, &material, &bcs, &options).unwrap();

        // 8 nodes * 3 DOFs = 24 DOFs
        assert_eq!(result.n_dofs, 24);

        // Stiffness matrix should have non-zeros
        assert!(result.stiffness.nnz() > 0);

        // Stiffness should be symmetric (use relative tolerance for large values)
        let dense = nalgebra::DMatrix::from(&result.stiffness);
        for i in 0..24 {
            for j in 0..24 {
                let kij = dense[(i, j)];
                let kji = dense[(j, i)];
                let max_abs = kij.abs().max(kji.abs()).max(1.0);
                assert!(
                    (kij - kji).abs() / max_abs < 1e-10,
                    "Stiffness not symmetric at ({}, {}): {} vs {} (rel diff: {:e})",
                    i,
                    j,
                    kij,
                    kji,
                    (kij - kji).abs() / max_abs
                );
            }
        }

        // Diagonal entries should be positive
        for i in 0..24 {
            assert!(dense[(i, i)] > 0.0, "Diagonal {} is not positive", i);
        }
    }

    #[test]
    fn test_multi_element_assembly() {
        // Create a mesh with two tetrahedra sharing a face
        let mut mesh = Mesh::new();

        // 5 nodes: 4 for first tet, 5th is apex of second tet
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0)); // 0
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0)); // 1
        mesh.add_node(Vector3::new(0.5, 1.0, 0.0)); // 2
        mesh.add_node(Vector3::new(0.5, 0.5, 1.0)); // 3
        mesh.add_node(Vector3::new(0.5, 0.5, -1.0)); // 4 (below the shared face)

        // First tet: nodes 0, 1, 2, 3
        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3])
            .unwrap();
        // Second tet: nodes 0, 1, 2, 4 (shares face 0-1-2)
        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 4])
            .unwrap();

        let material = Material::steel();
        let bcs: Vec<BoundaryCondition> = vec![];
        let options = AssemblyOptions::default();

        let result = assemble(&mesh, &material, &bcs, &options).unwrap();

        // 5 nodes * 3 DOFs = 15 DOFs
        assert_eq!(result.n_dofs, 15);

        // Stiffness matrix should have contributions from both elements
        // Shared nodes (0, 1, 2) should have accumulated stiffness
        let dense = nalgebra::DMatrix::from(&result.stiffness);

        // Check symmetry
        for i in 0..15 {
            for j in 0..15 {
                assert!(
                    (dense[(i, j)] - dense[(j, i)]).abs() < 1e-6,
                    "Stiffness not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_single_tri3_assembly() {
        // Create a single triangle mesh (2D plane stress)
        let mut mesh = Mesh::new();

        // Unit right triangle in XY plane
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));

        mesh.add_element(ElementType::Tri3, vec![0, 1, 2]).unwrap();

        let material = Material::steel();
        let bcs: Vec<BoundaryCondition> = vec![];
        let options = AssemblyOptions::default();

        let result = assemble(&mesh, &material, &bcs, &options).unwrap();

        // 3 nodes * 2 DOFs = 6 DOFs
        assert_eq!(result.n_dofs, 6);

        // Stiffness matrix should have non-zeros
        assert!(result.stiffness.nnz() > 0);

        // Check symmetry
        let dense = nalgebra::DMatrix::from(&result.stiffness);
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (dense[(i, j)] - dense[(j, i)]).abs() < 1e-6,
                    "Stiffness not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Diagonal entries should be positive
        for i in 0..6 {
            assert!(dense[(i, i)] > 0.0, "Diagonal {} is not positive", i);
        }
    }

    #[test]
    fn test_single_quad4_assembly() {
        // Create a single quad mesh (2D plane stress)
        let mut mesh = Mesh::new();

        // Unit square in XY plane
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 1.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));

        mesh.add_element(ElementType::Quad4, vec![0, 1, 2, 3])
            .unwrap();

        let material = Material::steel();
        let bcs: Vec<BoundaryCondition> = vec![];
        let options = AssemblyOptions::default();

        let result = assemble(&mesh, &material, &bcs, &options).unwrap();

        // 4 nodes * 2 DOFs = 8 DOFs
        assert_eq!(result.n_dofs, 8);

        // Stiffness matrix should have non-zeros
        assert!(result.stiffness.nnz() > 0);

        // Check symmetry
        let dense = nalgebra::DMatrix::from(&result.stiffness);
        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    (dense[(i, j)] - dense[(j, i)]).abs() < 1e-6,
                    "Stiffness not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Diagonal entries should be positive
        for i in 0..8 {
            assert!(dense[(i, i)] > 0.0, "Diagonal {} is not positive", i);
        }
    }

    #[test]
    fn test_2d_boundary_conditions() {
        // Create a triangle mesh and apply 2D BCs
        let mut mesh = Mesh::new();

        mesh.add_node(Vector3::new(0.0, 0.0, 0.0)); // Node 0
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0)); // Node 1
        mesh.add_node(Vector3::new(0.5, 1.0, 0.0)); // Node 2

        mesh.add_element(ElementType::Tri3, vec![0, 1, 2]).unwrap();

        let material = Material::steel();
        let bcs = vec![
            // Fix node 0 in both directions
            BoundaryCondition::Displacement {
                node: 0,
                dof: 0,
                value: 0.0,
            },
            BoundaryCondition::Displacement {
                node: 0,
                dof: 1,
                value: 0.0,
            },
            // Apply force at node 2 in y-direction
            BoundaryCondition::Force {
                node: 2,
                dof: 1,
                value: 1000.0,
            },
        ];
        let options = AssemblyOptions::default();

        let result = assemble(&mesh, &material, &bcs, &options).unwrap();

        // 3 nodes * 2 DOFs = 6 DOFs
        assert_eq!(result.n_dofs, 6);

        // Check constraints were recorded correctly
        // Node 0, dof 0 = global DOF 0
        // Node 0, dof 1 = global DOF 1
        assert!(result.constraints.contains_key(&0));
        assert!(result.constraints.contains_key(&1));

        // Check force was applied at node 2, dof 1 = global DOF 5
        assert!((result.rhs[5] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_tri3_assembly() {
        // Create a mesh with two triangles sharing an edge
        let mut mesh = Mesh::new();

        // 4 nodes forming a square split into two triangles
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0)); // 0
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0)); // 1
        mesh.add_node(Vector3::new(1.0, 1.0, 0.0)); // 2
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0)); // 3

        // Lower-right triangle
        mesh.add_element(ElementType::Tri3, vec![0, 1, 2]).unwrap();
        // Upper-left triangle
        mesh.add_element(ElementType::Tri3, vec![0, 2, 3]).unwrap();

        let material = Material::steel();
        let bcs: Vec<BoundaryCondition> = vec![];
        let options = AssemblyOptions::default();

        let result = assemble(&mesh, &material, &bcs, &options).unwrap();

        // 4 nodes * 2 DOFs = 8 DOFs
        assert_eq!(result.n_dofs, 8);

        // Check symmetry
        let dense = nalgebra::DMatrix::from(&result.stiffness);
        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    (dense[(i, j)] - dense[(j, i)]).abs() < 1e-6,
                    "Stiffness not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }

        // All diagonal entries should be positive
        for i in 0..8 {
            assert!(dense[(i, i)] > 0.0, "Diagonal {} is not positive", i);
        }

        // There should be coupling between nodes of each element
        // Element 1 (nodes 0,1,2): DOFs 0-5
        // Element 2 (nodes 0,2,3): DOFs 0-1, 4-7
        // Coupling should exist between nodes within each element
        // Node 0 (DOFs 0,1) and Node 1 (DOFs 2,3) from element 1
        assert!(
            dense[(0, 2)].abs() > 1e-10 || dense[(0, 3)].abs() > 1e-10,
            "Nodes 0 and 1 should have coupling"
        );
    }
}
