//! 10-node tetrahedron (Tet10) element.
//!
//! The Tet10 is a quadratic 3D solid element with:
//! - 4 nodes at vertices
//! - 6 nodes at edge midpoints
//! - 3 DOFs per node (u, v, w displacements)
//! - 30 total DOFs
//! - Quadratic shape functions
//! - 4-point Gauss integration
//!
//! # Shape Functions
//!
//! Quadratic shape functions in terms of barycentric coordinates (L1, L2, L3, L4):
//! - Corner nodes (1-4): N_i = L_i * (2*L_i - 1)
//! - Midside nodes (5-10): N_ij = 4 * L_i * L_j
//!
//! where L1 + L2 + L3 + L4 = 1.
//!
//! # Node Numbering
//!
//! ```text
//!              4
//!             /|\
//!            / | \
//!           /  |  \
//!          /   |   \
//!         8    |    10
//!        /     9     \
//!       /      |      \
//!      /       |       \
//!     1--------7--------3
//!      \      /        /
//!       \    /        /
//!        \  5        6
//!         \/        /
//!          2-------/
//!
//! Vertices:
//!   Node 1: (1, 0, 0, 0) - vertex at parametric origin
//!   Node 2: (0, 1, 0, 0)
//!   Node 3: (0, 0, 1, 0)
//!   Node 4: (0, 0, 0, 1)
//!
//! Edge midpoints:
//!   Node 5: midpoint of edge 1-2 = (0.5, 0.5, 0, 0)
//!   Node 6: midpoint of edge 2-3 = (0, 0.5, 0.5, 0)
//!   Node 7: midpoint of edge 1-3 = (0.5, 0, 0.5, 0)
//!   Node 8: midpoint of edge 1-4 = (0.5, 0, 0, 0.5)
//!   Node 9: midpoint of edge 2-4 = (0, 0.5, 0, 0.5)
//!   Node 10: midpoint of edge 3-4 = (0, 0, 0.5, 0.5)
//! ```
//!
//! # Advantages over Tet4
//!
//! - Linear strain variation within element
//! - Much better accuracy for bending
//! - Less susceptible to locking
//! - Standard element for production FEA

use crate::element::gauss::gauss_tet;
use crate::element::Element;
use crate::material::Material;
use crate::types::{Point3, StressTensor};
use nalgebra::{DMatrix, Matrix3, Vector3, Vector6};

/// 10-node tetrahedral element (quadratic tetrahedron).
#[derive(Debug, Clone, Copy, Default)]
pub struct Tet10;

impl Tet10 {
    /// Create a new Tet10 element.
    pub fn new() -> Self {
        Self
    }

    /// Compute shape functions at a point in barycentric coordinates.
    ///
    /// # Arguments
    ///
    /// * `l1`, `l2`, `l3`, `l4` - Barycentric coordinates (should sum to 1)
    ///
    /// # Returns
    ///
    /// Array of 10 shape function values [N_1, N_2, ..., N_10]
    #[cfg(test)]
    fn shape_functions(l1: f64, l2: f64, l3: f64, l4: f64) -> [f64; 10] {
        // Corner nodes: N_i = L_i * (2*L_i - 1)
        let n1 = l1 * (2.0 * l1 - 1.0);
        let n2 = l2 * (2.0 * l2 - 1.0);
        let n3 = l3 * (2.0 * l3 - 1.0);
        let n4 = l4 * (2.0 * l4 - 1.0);

        // Midside nodes: N_ij = 4 * L_i * L_j
        let n5 = 4.0 * l1 * l2;  // Edge 1-2
        let n6 = 4.0 * l2 * l3;  // Edge 2-3
        let n7 = 4.0 * l1 * l3;  // Edge 1-3
        let n8 = 4.0 * l1 * l4;  // Edge 1-4
        let n9 = 4.0 * l2 * l4;  // Edge 2-4
        let n10 = 4.0 * l3 * l4; // Edge 3-4

        [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]
    }

    /// Compute shape function derivatives with respect to barycentric coordinates.
    ///
    /// Returns dN/dL for each of the 4 barycentric coordinates and 10 shape functions.
    /// Result is organized as [[dN1/dL1, dN2/dL1, ...], [dN1/dL2, ...], ...]
    fn shape_derivatives_barycentric(l1: f64, l2: f64, l3: f64, l4: f64) -> [[f64; 10]; 4] {
        // dN_corner/dL_i = 4*L_i - 1 if i matches, 0 otherwise
        // dN_midside/dL_i = 4*L_j if edge connects i-j, 0 otherwise

        // Derivatives with respect to L1
        let dn_dl1 = [
            4.0 * l1 - 1.0,  // dN1/dL1: corner node 1
            0.0,             // dN2/dL1
            0.0,             // dN3/dL1
            0.0,             // dN4/dL1
            4.0 * l2,        // dN5/dL1: edge 1-2
            0.0,             // dN6/dL1
            4.0 * l3,        // dN7/dL1: edge 1-3
            4.0 * l4,        // dN8/dL1: edge 1-4
            0.0,             // dN9/dL1
            0.0,             // dN10/dL1
        ];

        // Derivatives with respect to L2
        let dn_dl2 = [
            0.0,             // dN1/dL2
            4.0 * l2 - 1.0,  // dN2/dL2: corner node 2
            0.0,             // dN3/dL2
            0.0,             // dN4/dL2
            4.0 * l1,        // dN5/dL2: edge 1-2
            4.0 * l3,        // dN6/dL2: edge 2-3
            0.0,             // dN7/dL2
            0.0,             // dN8/dL2
            4.0 * l4,        // dN9/dL2: edge 2-4
            0.0,             // dN10/dL2
        ];

        // Derivatives with respect to L3
        let dn_dl3 = [
            0.0,             // dN1/dL3
            0.0,             // dN2/dL3
            4.0 * l3 - 1.0,  // dN3/dL3: corner node 3
            0.0,             // dN4/dL3
            0.0,             // dN5/dL3
            4.0 * l2,        // dN6/dL3: edge 2-3
            4.0 * l1,        // dN7/dL3: edge 1-3
            0.0,             // dN8/dL3
            0.0,             // dN9/dL3
            4.0 * l4,        // dN10/dL3: edge 3-4
        ];

        // Derivatives with respect to L4
        let dn_dl4 = [
            0.0,             // dN1/dL4
            0.0,             // dN2/dL4
            0.0,             // dN3/dL4
            4.0 * l4 - 1.0,  // dN4/dL4: corner node 4
            0.0,             // dN5/dL4
            0.0,             // dN6/dL4
            0.0,             // dN7/dL4
            4.0 * l1,        // dN8/dL4: edge 1-4
            4.0 * l2,        // dN9/dL4: edge 2-4
            4.0 * l3,        // dN10/dL4: edge 3-4
        ];

        [dn_dl1, dn_dl2, dn_dl3, dn_dl4]
    }

    /// Compute the Jacobian matrix from parametric to physical coordinates.
    ///
    /// For a tetrahedron, we use the parametric coordinates (ξ, η, ζ) where:
    /// L1 = 1 - ξ - η - ζ, L2 = ξ, L3 = η, L4 = ζ
    ///
    /// The Jacobian maps:
    /// [dx, dy, dz]^T = J * [dξ, dη, dζ]^T
    fn jacobian(coords: &[Point3], l1: f64, l2: f64, l3: f64, l4: f64) -> Matrix3<f64> {
        let dn_dl = Self::shape_derivatives_barycentric(l1, l2, l3, l4);

        // Convert from barycentric derivatives to parametric derivatives
        // Since L1 = 1 - ξ - η - ζ:
        // dN/dξ = dN/dL2 - dN/dL1 (because ∂L2/∂ξ = 1, ∂L1/∂ξ = -1)
        // dN/dη = dN/dL3 - dN/dL1
        // dN/dζ = dN/dL4 - dN/dL1
        let mut dn_dxi = [0.0; 10];
        let mut dn_deta = [0.0; 10];
        let mut dn_dzeta = [0.0; 10];

        for i in 0..10 {
            dn_dxi[i] = dn_dl[1][i] - dn_dl[0][i];   // dN/dξ = dN/dL2 - dN/dL1
            dn_deta[i] = dn_dl[2][i] - dn_dl[0][i];  // dN/dη = dN/dL3 - dN/dL1
            dn_dzeta[i] = dn_dl[3][i] - dn_dl[0][i]; // dN/dζ = dN/dL4 - dN/dL1
        }

        // Compute Jacobian: J_ij = Σ_k (dN_k/dξ_j * x_k[i])
        // J = [∂x/∂ξ  ∂y/∂ξ  ∂z/∂ξ]
        //     [∂x/∂η  ∂y/∂η  ∂z/∂η]
        //     [∂x/∂ζ  ∂y/∂ζ  ∂z/∂ζ]
        let mut j = Matrix3::zeros();

        for i in 0..10 {
            let x = coords[i][0];
            let y = coords[i][1];
            let z = coords[i][2];

            // Row 0: ∂(x,y,z)/∂ξ
            j[(0, 0)] += dn_dxi[i] * x;
            j[(0, 1)] += dn_dxi[i] * y;
            j[(0, 2)] += dn_dxi[i] * z;

            // Row 1: ∂(x,y,z)/∂η
            j[(1, 0)] += dn_deta[i] * x;
            j[(1, 1)] += dn_deta[i] * y;
            j[(1, 2)] += dn_deta[i] * z;

            // Row 2: ∂(x,y,z)/∂ζ
            j[(2, 0)] += dn_dzeta[i] * x;
            j[(2, 1)] += dn_dzeta[i] * y;
            j[(2, 2)] += dn_dzeta[i] * z;
        }

        j
    }

    /// Compute the B-matrix (strain-displacement) at a point.
    ///
    /// Returns the 6×30 B-matrix and the Jacobian determinant.
    fn compute_b_matrix(coords: &[Point3], l1: f64, l2: f64, l3: f64, l4: f64) -> (DMatrix<f64>, f64) {
        let dn_dl = Self::shape_derivatives_barycentric(l1, l2, l3, l4);

        // Convert to parametric derivatives
        let mut dn_dxi = [0.0; 10];
        let mut dn_deta = [0.0; 10];
        let mut dn_dzeta = [0.0; 10];

        for i in 0..10 {
            dn_dxi[i] = dn_dl[1][i] - dn_dl[0][i];
            dn_deta[i] = dn_dl[2][i] - dn_dl[0][i];
            dn_dzeta[i] = dn_dl[3][i] - dn_dl[0][i];
        }

        let j = Self::jacobian(coords, l1, l2, l3, l4);
        let det_j = j.determinant();

        assert!(
            det_j > 0.0,
            "Negative Jacobian determinant ({}) indicates inverted element",
            det_j
        );

        let j_inv = j.try_inverse().expect("Jacobian is singular");

        // Compute physical derivatives: [dN/dx, dN/dy, dN/dz]^T = J^(-1) * [dN/dξ, dN/dη, dN/dζ]^T
        let mut dn_dx = [0.0; 10];
        let mut dn_dy = [0.0; 10];
        let mut dn_dz = [0.0; 10];

        for i in 0..10 {
            let dn_dnat = Vector3::new(dn_dxi[i], dn_deta[i], dn_dzeta[i]);
            let dn_dphys = j_inv * dn_dnat;
            dn_dx[i] = dn_dphys[0];
            dn_dy[i] = dn_dphys[1];
            dn_dz[i] = dn_dphys[2];
        }

        // Construct B-matrix (6×30)
        // ε = [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz]^T = B * u
        let mut b = DMatrix::zeros(6, 30);

        for i in 0..10 {
            let col = i * 3; // Starting column for node i

            // ε_xx = ∂u/∂x
            b[(0, col)] = dn_dx[i];

            // ε_yy = ∂v/∂y
            b[(1, col + 1)] = dn_dy[i];

            // ε_zz = ∂w/∂z
            b[(2, col + 2)] = dn_dz[i];

            // γ_xy = ∂u/∂y + ∂v/∂x
            b[(3, col)] = dn_dy[i];
            b[(3, col + 1)] = dn_dx[i];

            // γ_yz = ∂v/∂z + ∂w/∂y
            b[(4, col + 1)] = dn_dz[i];
            b[(4, col + 2)] = dn_dy[i];

            // γ_xz = ∂u/∂z + ∂w/∂x
            b[(5, col)] = dn_dz[i];
            b[(5, col + 2)] = dn_dx[i];
        }

        (b, det_j)
    }
}

impl Element for Tet10 {
    fn n_nodes(&self) -> usize {
        10
    }

    fn dofs_per_node(&self) -> usize {
        3
    }

    fn stiffness(&self, coords: &[Point3], material: &Material) -> DMatrix<f64> {
        assert_eq!(
            coords.len(),
            10,
            "Tet10 requires exactly 10 nodal coordinates"
        );

        let d = material.constitutive_3d();
        let mut k = DMatrix::zeros(30, 30);

        // 4-point tetrahedral Gauss integration
        let gauss_points = gauss_tet(4);

        for gp in &gauss_points {
            // Gauss points are in barycentric coordinates [L1, L2, L3, L4]
            let l1 = gp.coords[0];
            let l2 = gp.coords[1];
            let l3 = gp.coords[2];
            let l4 = gp.coords[3];
            let w = gp.weight;

            let (b, det_j) = Self::compute_b_matrix(coords, l1, l2, l3, l4);

            // K += B^T * D * B * |J| * w
            // Gauss weights for tetrahedral quadrature already include the 1/6 volume factor
            let db = &d * &b; // 6×30
            let btdb = b.transpose() * db; // 30×30

            k += btdb * det_j * w;
        }

        k
    }

    fn stress(
        &self,
        coords: &[Point3],
        displacements: &[f64],
        material: &Material,
    ) -> Vec<StressTensor> {
        assert_eq!(
            coords.len(),
            10,
            "Tet10 requires exactly 10 nodal coordinates"
        );
        assert_eq!(
            displacements.len(),
            30,
            "Tet10 requires 30 displacement DOFs"
        );

        let d = material.constitutive_3d();
        let u = nalgebra::DVector::from_row_slice(displacements);

        // Compute stress at each integration point
        let gauss_points = gauss_tet(4);
        let mut stresses = Vec::with_capacity(gauss_points.len());

        for gp in &gauss_points {
            let l1 = gp.coords[0];
            let l2 = gp.coords[1];
            let l3 = gp.coords[2];
            let l4 = gp.coords[3];

            let (b, _det_j) = Self::compute_b_matrix(coords, l1, l2, l3, l4);

            // ε = B * u
            let strain = &b * &u; // 6×1

            // σ = D * ε
            let stress_vec = &d * Vector6::from_iterator(strain.iter().cloned());

            stresses.push(StressTensor(stress_vec));
        }

        stresses
    }

    fn volume(&self, coords: &[Point3]) -> f64 {
        assert_eq!(
            coords.len(),
            10,
            "Tet10 requires exactly 10 nodal coordinates"
        );

        let gauss_points = gauss_tet(4);
        let mut volume = 0.0;

        for gp in &gauss_points {
            let l1 = gp.coords[0];
            let l2 = gp.coords[1];
            let l3 = gp.coords[2];
            let l4 = gp.coords[3];
            let w = gp.weight;

            let j = Self::jacobian(coords, l1, l2, l3, l4);
            let det_j = j.determinant();

            // Gauss weights already include the 1/6 factor
            volume += det_j * w;
        }

        volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Create a regular unit tetrahedron with midside nodes.
    /// Vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    fn unit_tet10() -> Vec<Point3> {
        vec![
            // Corner nodes (1-4)
            Point3::new(0.0, 0.0, 0.0), // Node 1 (index 0)
            Point3::new(1.0, 0.0, 0.0), // Node 2 (index 1)
            Point3::new(0.0, 1.0, 0.0), // Node 3 (index 2)
            Point3::new(0.0, 0.0, 1.0), // Node 4 (index 3)
            // Midside nodes (5-10)
            Point3::new(0.5, 0.0, 0.0), // Node 5: midpoint of edge 1-2 (index 4)
            Point3::new(0.5, 0.5, 0.0), // Node 6: midpoint of edge 2-3 (index 5)
            Point3::new(0.0, 0.5, 0.0), // Node 7: midpoint of edge 1-3 (index 6)
            Point3::new(0.0, 0.0, 0.5), // Node 8: midpoint of edge 1-4 (index 7)
            Point3::new(0.5, 0.0, 0.5), // Node 9: midpoint of edge 2-4 (index 8)
            Point3::new(0.0, 0.5, 0.5), // Node 10: midpoint of edge 3-4 (index 9)
        ]
    }

    /// Create a scaled tetrahedron (2x in each direction).
    fn scaled_tet10() -> Vec<Point3> {
        unit_tet10().iter().map(|p| p * 2.0).collect()
    }

    #[test]
    fn test_tet10_shape_functions_sum_to_one() {
        // Shape functions should sum to 1 at any point
        let test_points = [
            (0.25, 0.25, 0.25, 0.25), // centroid
            (1.0, 0.0, 0.0, 0.0),     // corner 1
            (0.0, 1.0, 0.0, 0.0),     // corner 2
            (0.5, 0.5, 0.0, 0.0),     // midside node 5
            (0.1, 0.2, 0.3, 0.4),     // arbitrary interior
        ];

        for (l1, l2, l3, l4) in test_points {
            let n = Tet10::shape_functions(l1, l2, l3, l4);
            let sum: f64 = n.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_tet10_shape_functions_at_corners() {
        // At corner nodes, only the corresponding corner shape function should be 1
        let corners = [
            (1.0, 0.0, 0.0, 0.0), // L1 = 1
            (0.0, 1.0, 0.0, 0.0), // L2 = 1
            (0.0, 0.0, 1.0, 0.0), // L3 = 1
            (0.0, 0.0, 0.0, 1.0), // L4 = 1
        ];

        for (corner_idx, &(l1, l2, l3, l4)) in corners.iter().enumerate() {
            let n = Tet10::shape_functions(l1, l2, l3, l4);

            // Corner shape function should be 1
            assert_relative_eq!(n[corner_idx], 1.0, epsilon = 1e-14);

            // All other shape functions should be 0
            for (i, &val) in n.iter().enumerate() {
                if i != corner_idx {
                    assert_relative_eq!(val, 0.0, epsilon = 1e-14);
                }
            }
        }
    }

    #[test]
    fn test_tet10_shape_functions_at_midside() {
        // At midside node 5 (L1=L2=0.5, L3=L4=0), N5 = 4*0.5*0.5 = 1
        let n = Tet10::shape_functions(0.5, 0.5, 0.0, 0.0);
        assert_relative_eq!(n[4], 1.0, epsilon = 1e-14); // N5
        assert_relative_eq!(n[0], 0.0, epsilon = 1e-14); // N1 = 0.5*(2*0.5-1) = 0
        assert_relative_eq!(n[1], 0.0, epsilon = 1e-14); // N2 = 0

        // At midside node 6 (L2=L3=0.5)
        let n = Tet10::shape_functions(0.0, 0.5, 0.5, 0.0);
        assert_relative_eq!(n[5], 1.0, epsilon = 1e-14); // N6
    }

    #[test]
    fn test_tet10_volume_unit() {
        let tet = Tet10::new();
        let coords = unit_tet10();
        let vol = tet.volume(&coords);
        // Unit tetrahedron volume = 1/6
        assert_relative_eq!(vol, 1.0 / 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_tet10_volume_scaled() {
        let tet = Tet10::new();
        let coords = scaled_tet10();
        let vol = tet.volume(&coords);
        // Scaled by 2 in each direction -> volume * 8
        assert_relative_eq!(vol, 8.0 / 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_tet10_node_count() {
        let tet = Tet10::new();
        assert_eq!(tet.n_nodes(), 10);
        assert_eq!(tet.dofs_per_node(), 3);
        assert_eq!(tet.n_dofs(), 30);
    }

    #[test]
    fn test_tet10_stiffness_symmetric() {
        let tet = Tet10::new();
        let coords = unit_tet10();
        let mat = Material::steel();

        let k = tet.stiffness(&coords, &mat);

        assert_eq!(k.nrows(), 30);
        assert_eq!(k.ncols(), 30);

        // Find the maximum absolute value to scale the tolerance
        let k_max = k.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        for i in 0..30 {
            for j in 0..30 {
                // Use absolute tolerance scaled by matrix magnitude
                let diff = (k[(i, j)] - k[(j, i)]).abs();
                assert!(
                    diff < k_max * 1e-12,
                    "K[{},{}] = {} != K[{},{}] = {}, diff = {}, k_max = {}",
                    i, j, k[(i, j)], j, i, k[(j, i)], diff, k_max
                );
            }
        }
    }

    #[test]
    fn test_tet10_stiffness_positive_diagonal() {
        let tet = Tet10::new();
        let coords = unit_tet10();
        let mat = Material::steel();

        let k = tet.stiffness(&coords, &mat);

        for i in 0..30 {
            assert!(
                k[(i, i)] > 0.0,
                "K[{},{}] = {} should be positive",
                i,
                i,
                k[(i, i)]
            );
        }
    }

    #[test]
    fn test_tet10_rigid_body_translation() {
        let tet = Tet10::new();
        let coords = unit_tet10();
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = tet.stiffness(&coords, &mat);

        // x-translation: all 10 nodes move by (1, 0, 0)
        let mut u_x = vec![0.0; 30];
        for i in 0..10 {
            u_x[i * 3] = 1.0; // ux = 1 for all nodes
        }
        let u_x = nalgebra::DVector::from_vec(u_x);
        let f_x = &k * &u_x;
        assert_relative_eq!(f_x.norm(), 0.0, epsilon = 1e-10);

        // y-translation
        let mut u_y = vec![0.0; 30];
        for i in 0..10 {
            u_y[i * 3 + 1] = 1.0;
        }
        let u_y = nalgebra::DVector::from_vec(u_y);
        let f_y = &k * &u_y;
        assert_relative_eq!(f_y.norm(), 0.0, epsilon = 1e-10);

        // z-translation
        let mut u_z = vec![0.0; 30];
        for i in 0..10 {
            u_z[i * 3 + 2] = 1.0;
        }
        let u_z = nalgebra::DVector::from_vec(u_z);
        let f_z = &k * &u_z;
        assert_relative_eq!(f_z.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tet10_linear_strain_patch() {
        // For a quadratic element, verify constant strain produces correct stress
        let tet = Tet10::new();
        let coords = unit_tet10();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Impose uniform ε_xx = 0.001 by linear displacement u = 0.001 * x
        let strain_xx = 0.001;
        let mut displacements = [0.0; 30];

        for i in 0..10 {
            // u = ε_xx * x, v = 0, w = 0
            displacements[i * 3] = strain_xx * coords[i][0];
        }

        let stresses = tet.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 4); // 4 integration points

        let d = mat.constitutive_3d();
        let expected_sigma_xx = d[(0, 0)] * strain_xx;
        let expected_sigma_yy = d[(1, 0)] * strain_xx;
        let expected_sigma_zz = d[(2, 0)] * strain_xx;

        for stress in &stresses {
            assert_relative_eq!(
                stress.0[0],
                expected_sigma_xx,
                epsilon = 1e-6,
                max_relative = 1e-6
            );
            assert_relative_eq!(
                stress.0[1],
                expected_sigma_yy,
                epsilon = 1e-6,
                max_relative = 1e-6
            );
            assert_relative_eq!(
                stress.0[2],
                expected_sigma_zz,
                epsilon = 1e-6,
                max_relative = 1e-6
            );
            // Shear stresses should be zero
            assert_relative_eq!(stress.0[3], 0.0, epsilon = 1e-10);
            assert_relative_eq!(stress.0[4], 0.0, epsilon = 1e-10);
            assert_relative_eq!(stress.0[5], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tet10_stress_count() {
        let tet = Tet10::new();
        let coords = unit_tet10();
        let mat = Material::steel();
        let displacements = [0.0; 30];

        let stresses = tet.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 4); // One stress per integration point
    }

    #[test]
    fn test_tet10_quadratic_geometry() {
        // Test that the element handles curved (quadratic) geometry correctly
        // by moving a midside node off the straight edge
        let tet = Tet10::new();
        let mut coords = unit_tet10();

        // Move midside node 5 (edge 1-2) slightly perpendicular to the edge
        coords[4] = Point3::new(0.5, 0.1, 0.0); // Was (0.5, 0.0, 0.0)

        // The element should still work (volume might be slightly different)
        let vol = tet.volume(&coords);
        assert!(vol > 0.0, "Volume should be positive");

        // Stiffness should still be symmetric
        let mat = Material::steel();
        let k = tet.stiffness(&coords, &mat);

        // Find the maximum absolute value to scale the tolerance
        let k_max = k.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        for i in 0..30 {
            for j in 0..30 {
                let diff = (k[(i, j)] - k[(j, i)]).abs();
                assert!(
                    diff < k_max * 1e-10,
                    "K[{},{}] != K[{},{}], diff = {}, k_max = {}",
                    i, j, j, i, diff, k_max
                );
            }
        }
    }
}
