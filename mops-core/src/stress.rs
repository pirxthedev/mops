//! Stress recovery from displacement solution.
//!
//! After solving Ku = f for displacements, this module computes element stresses.
//! The stress recovery pipeline:
//! 1. Loop through all elements
//! 2. Extract element nodal displacements from global solution
//! 3. Compute strain: ε = B * u_e (strain-displacement relation)
//! 4. Compute stress: σ = D * ε (constitutive relation)
//!
//! Stresses are computed at element integration points (Gauss points).
//! For nodal stress output, averaging from adjacent elements is typically performed.

use crate::element::create_element;
use crate::material::Material;
use crate::mesh::Mesh;
use crate::types::StressTensor;
use rayon::prelude::*;

/// Stress result for a single element.
#[derive(Debug, Clone)]
pub struct ElementStress {
    /// Element index in the mesh.
    pub element_id: usize,
    /// Stress tensors at each integration point.
    pub integration_point_stresses: Vec<StressTensor>,
}

impl ElementStress {
    /// Compute the average stress across all integration points.
    pub fn average_stress(&self) -> StressTensor {
        if self.integration_point_stresses.is_empty() {
            return StressTensor::zero();
        }
        let n = self.integration_point_stresses.len() as f64;
        let mut sum = [0.0; 6];
        for s in &self.integration_point_stresses {
            for i in 0..6 {
                sum[i] += s.0[i];
            }
        }
        for s in &mut sum {
            *s /= n;
        }
        StressTensor::new(sum)
    }

    /// Maximum von Mises stress among all integration points.
    pub fn max_von_mises(&self) -> f64 {
        self.integration_point_stresses
            .iter()
            .map(|s| s.von_mises())
            .fold(0.0, f64::max)
    }
}

/// Stress recovery results for the entire mesh.
#[derive(Debug, Clone)]
pub struct StressField {
    /// Element stresses indexed by element ID.
    pub element_stresses: Vec<ElementStress>,
}

impl StressField {
    /// Get stress for a specific element.
    pub fn element(&self, elem_id: usize) -> Option<&ElementStress> {
        self.element_stresses.get(elem_id)
    }

    /// Maximum von Mises stress across all elements.
    pub fn max_von_mises(&self) -> f64 {
        self.element_stresses
            .iter()
            .map(|es| es.max_von_mises())
            .fold(0.0, f64::max)
    }

    /// Compute average element stresses.
    pub fn average_stresses(&self) -> Vec<StressTensor> {
        self.element_stresses
            .iter()
            .map(|es| es.average_stress())
            .collect()
    }

    /// Number of elements with stress data.
    pub fn n_elements(&self) -> usize {
        self.element_stresses.len()
    }

    /// Get all von Mises stresses (one per element, averaged).
    pub fn von_mises_stresses(&self) -> Vec<f64> {
        self.element_stresses
            .iter()
            .map(|es| es.average_stress().von_mises())
            .collect()
    }
}

/// Recover stresses from displacement solution.
///
/// This is the main entry point for stress recovery. Given a mesh, material,
/// and the solved displacement vector, it computes stresses at each element's
/// integration points.
///
/// # Arguments
///
/// * `mesh` - The finite element mesh
/// * `material` - Material properties (uniform for all elements)
/// * `displacements` - Global displacement vector from solver (n_nodes * 3)
///
/// # Returns
///
/// `StressField` containing stresses for all elements.
///
/// # Example
///
/// ```ignore
/// let system = assemble(&mesh, &material, &bcs, &options)?;
/// let displacements = solver.solve(&system.stiffness, &system.rhs)?;
/// let stresses = recover_stresses(&mesh, &material, &displacements);
/// println!("Max von Mises stress: {:.2} Pa", stresses.max_von_mises());
/// ```
///
/// # Note
///
/// For multi-material analysis, use `recover_stresses_with_materials` instead.
pub fn recover_stresses(mesh: &Mesh, material: &Material, displacements: &[f64]) -> StressField {
    // Get DOFs per node from mesh (2 for 2D elements, 3 for 3D elements)
    let dofs_per_node = mesh.dofs_per_node().unwrap_or(3);

    // Parallel stress recovery over elements
    let element_stresses: Vec<ElementStress> = mesh
        .elements()
        .par_iter()
        .enumerate()
        .map(|(elem_idx, connectivity)| {
            // Create element implementation
            let element = create_element(connectivity.element_type);

            // Get DOFs per node for this specific element type
            let elem_dofs_per_node = connectivity.element_type.dofs_per_node();

            // Get element nodal coordinates
            let coords = mesh.element_coords(elem_idx).expect("Valid element index");

            // Extract element nodal displacements
            let n_nodes = connectivity.nodes.len();
            let n_dofs = n_nodes * elem_dofs_per_node;
            let mut elem_displacements = vec![0.0; n_dofs];

            for (local_node, &global_node) in connectivity.nodes.iter().enumerate() {
                for dof in 0..elem_dofs_per_node {
                    let global_dof = global_node * dofs_per_node + dof;
                    let local_dof = local_node * elem_dofs_per_node + dof;
                    elem_displacements[local_dof] =
                        displacements.get(global_dof).copied().unwrap_or(0.0);
                }
            }

            // Compute stress at integration points
            let stresses = element.stress(&coords, &elem_displacements, material);

            ElementStress {
                element_id: elem_idx,
                integration_point_stresses: stresses,
            }
        })
        .collect();

    StressField { element_stresses }
}

/// Recover stresses with per-element material assignments.
///
/// This version allows stress recovery when different elements have different
/// materials assigned. The correct constitutive relationship is used for each
/// element to compute accurate stresses.
///
/// # Arguments
///
/// * `mesh` - The finite element mesh
/// * `materials` - List of materials that were used in assembly
/// * `element_materials` - Map from element index to material index in `materials`
///                         Elements not in the map use material index 0
/// * `displacements` - Global displacement vector from solver
///
/// # Returns
///
/// `StressField` containing stresses for all elements.
pub fn recover_stresses_with_materials(
    mesh: &Mesh,
    materials: &[Material],
    element_materials: &std::collections::HashMap<usize, usize>,
    displacements: &[f64],
) -> StressField {
    // Get DOFs per node from mesh (2 for 2D elements, 3 for 3D elements)
    let dofs_per_node = mesh.dofs_per_node().unwrap_or(3);

    // Parallel stress recovery over elements
    let element_stresses: Vec<ElementStress> = mesh
        .elements()
        .par_iter()
        .enumerate()
        .map(|(elem_idx, connectivity)| {
            // Create element implementation
            let element = create_element(connectivity.element_type);

            // Get DOFs per node for this specific element type
            let elem_dofs_per_node = connectivity.element_type.dofs_per_node();

            // Get element nodal coordinates
            let coords = mesh.element_coords(elem_idx).expect("Valid element index");

            // Extract element nodal displacements
            let n_nodes = connectivity.nodes.len();
            let n_dofs = n_nodes * elem_dofs_per_node;
            let mut elem_displacements = vec![0.0; n_dofs];

            for (local_node, &global_node) in connectivity.nodes.iter().enumerate() {
                for dof in 0..elem_dofs_per_node {
                    let global_dof = global_node * dofs_per_node + dof;
                    let local_dof = local_node * elem_dofs_per_node + dof;
                    elem_displacements[local_dof] =
                        displacements.get(global_dof).copied().unwrap_or(0.0);
                }
            }

            // Look up material for this element (default to first material)
            let material_idx = element_materials.get(&elem_idx).copied().unwrap_or(0);
            let material = &materials[material_idx.min(materials.len() - 1)];

            // Compute stress at integration points using element-specific material
            let stresses = element.stress(&coords, &elem_displacements, material);

            ElementStress {
                element_id: elem_idx,
                integration_point_stresses: stresses,
            }
        })
        .collect();

    StressField { element_stresses }
}

/// Compute nodal stresses by averaging contributions from adjacent elements.
///
/// This performs a simple averaging scheme where each node's stress is the
/// average of all elements that contain that node, weighted equally.
///
/// For more sophisticated stress smoothing (e.g., superconvergent patch recovery),
/// specialized post-processing methods would be used.
///
/// # Arguments
///
/// * `mesh` - The finite element mesh
/// * `stress_field` - Element stress data from `recover_stresses`
///
/// # Returns
///
/// Vector of nodal stress tensors (length = n_nodes)
pub fn compute_nodal_stresses(mesh: &Mesh, stress_field: &StressField) -> Vec<StressTensor> {
    let n_nodes = mesh.n_nodes();
    let mut nodal_sums = vec![[0.0; 6]; n_nodes];
    let mut nodal_counts = vec![0usize; n_nodes];

    // Accumulate element stress contributions to nodes
    for (elem_idx, connectivity) in mesh.elements().iter().enumerate() {
        if let Some(elem_stress) = stress_field.element(elem_idx) {
            let avg_stress = elem_stress.average_stress();
            for &node in &connectivity.nodes {
                for i in 0..6 {
                    nodal_sums[node][i] += avg_stress.0[i];
                }
                nodal_counts[node] += 1;
            }
        }
    }

    // Compute averages
    nodal_sums
        .into_iter()
        .zip(nodal_counts)
        .map(|(mut sum, count)| {
            if count > 0 {
                let c = count as f64;
                for s in &mut sum {
                    *s /= c;
                }
            }
            StressTensor::new(sum)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ElementType;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    #[test]
    fn test_stress_recovery_single_tet4() {
        // Create a single tet4 element
        let mut mesh = Mesh::new();
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3])
            .unwrap();

        let material = Material::steel();

        // Zero displacement -> zero stress
        let displacements = vec![0.0; 12];
        let stresses = recover_stresses(&mesh, &material, &displacements);

        assert_eq!(stresses.n_elements(), 1);
        let elem_stress = stresses.element(0).unwrap();
        assert_eq!(elem_stress.integration_point_stresses.len(), 1); // Tet4 has 1 int point
        assert_relative_eq!(elem_stress.max_von_mises(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stress_recovery_uniform_extension() {
        // Create a single tet4 and apply uniform extension in x
        let mut mesh = Mesh::new();
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3])
            .unwrap();

        let material = Material::steel();
        let e = material.youngs_modulus;
        let nu = material.poissons_ratio;

        // Uniform strain: ε_xx = 0.001 (0.1% extension)
        // Displacements: u = ε_xx * x, v = 0, w = 0
        // Node 0: (0,0,0) -> (0, 0, 0)
        // Node 1: (1,0,0) -> (0.001, 0, 0)
        // Node 2: (0,1,0) -> (0, 0, 0)
        // Node 3: (0,0,1) -> (0, 0, 0)
        let strain = 0.001;
        let displacements = vec![
            0.0, 0.0, 0.0, // Node 0
            strain, 0.0, 0.0, // Node 1
            0.0, 0.0, 0.0, // Node 2
            0.0, 0.0, 0.0, // Node 3
        ];

        let stresses = recover_stresses(&mesh, &material, &displacements);
        let elem_stress = stresses.element(0).unwrap();
        let avg_stress = elem_stress.average_stress();

        // For isotropic material under uniaxial strain:
        // σ_xx = E / (1+ν)(1-2ν) * [(1-ν)*ε_xx + ν*(ε_yy + ε_zz)]
        // With ε_yy = ε_zz = 0:
        // σ_xx = E * (1-ν) / ((1+ν)(1-2ν)) * ε_xx
        let expected_sigma_xx = e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu)) * strain;

        // σ_yy = σ_zz = E * ν / ((1+ν)(1-2ν)) * ε_xx
        let expected_sigma_yy = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) * strain;

        assert_relative_eq!(avg_stress.0[0], expected_sigma_xx, epsilon = 1e-6);
        assert_relative_eq!(avg_stress.0[1], expected_sigma_yy, epsilon = 1e-6);
        assert_relative_eq!(avg_stress.0[2], expected_sigma_yy, epsilon = 1e-6);

        // Shear stresses should be zero
        assert_relative_eq!(avg_stress.0[3], 0.0, epsilon = 1e-10);
        assert_relative_eq!(avg_stress.0[4], 0.0, epsilon = 1e-10);
        assert_relative_eq!(avg_stress.0[5], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nodal_stress_averaging() {
        // Two tet4 elements sharing nodes
        let mut mesh = Mesh::new();
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0)); // 0
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0)); // 1
        mesh.add_node(Vector3::new(0.5, 1.0, 0.0)); // 2
        mesh.add_node(Vector3::new(0.5, 0.5, 1.0)); // 3
        mesh.add_node(Vector3::new(0.5, 0.5, -1.0)); // 4

        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3])
            .unwrap();
        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 4])
            .unwrap();

        let material = Material::steel();

        // Non-zero displacement
        let displacements = vec![0.0; 15];
        let stresses = recover_stresses(&mesh, &material, &displacements);

        assert_eq!(stresses.n_elements(), 2);

        // Compute nodal stresses
        let nodal_stresses = compute_nodal_stresses(&mesh, &stresses);
        assert_eq!(nodal_stresses.len(), 5);

        // Nodes 0, 1, 2 are shared by both elements
        // Nodes 3, 4 are exclusive to one element each
    }

    #[test]
    fn test_von_mises_stresses() {
        let mut mesh = Mesh::new();
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3])
            .unwrap();

        let material = Material::steel();
        let displacements = vec![0.0; 12];
        let stresses = recover_stresses(&mesh, &material, &displacements);

        let vm = stresses.von_mises_stresses();
        assert_eq!(vm.len(), 1);
        assert_relative_eq!(vm[0], 0.0, epsilon = 1e-10);
    }
}
