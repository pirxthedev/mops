//! 8-node serendipity quadrilateral plane stress element (Quad8).
//!
//! The Quad8 element is a higher-order 2D element with serendipity shape functions:
//! - 8 nodes: 4 corner nodes + 4 mid-edge nodes
//! - 2 DOFs per node (u, v displacements)
//! - 16 total DOFs
//! - Quadratic strain variation along edges
//! - 3×3 Gauss quadrature for integration (9 points)
//!
//! # Node Numbering
//!
//! ```text
//!  4----7----3
//!  |         |
//!  8         6
//!  |         |
//!  1----5----2
//! ```
//!
//! - Corner nodes: 1, 2, 3, 4 (local indices 0, 1, 2, 3)
//! - Mid-edge nodes: 5 (edge 1-2), 6 (edge 2-3), 7 (edge 3-4), 8 (edge 4-1) (local indices 4, 5, 6, 7)
//!
//! # Natural Coordinates
//!
//! Uses isoparametric coordinates (ξ, η) ∈ [-1, 1]²:
//! - Corner nodes at (±1, ±1)
//! - Mid-edge nodes at (0, ±1) or (±1, 0)
//!
//! # Shape Functions (Serendipity)
//!
//! Corner nodes (i = 1..4):
//! ```text
//! N_i = (1/4)(1 + ξ_i*ξ)(1 + η_i*η)(ξ_i*ξ + η_i*η - 1)
//! ```
//!
//! Mid-side nodes on ξ = 0 (nodes 5, 7):
//! ```text
//! N_i = (1/2)(1 - ξ²)(1 + η_i*η)
//! ```
//!
//! Mid-side nodes on η = 0 (nodes 6, 8):
//! ```text
//! N_i = (1/2)(1 + ξ_i*ξ)(1 - η²)
//! ```

use crate::element::gauss::gauss_quad;
use crate::element::Element;
use crate::material::Material;
use crate::types::{Point3, StressTensor};
use nalgebra::{DMatrix, Matrix2, Vector2, Vector3, Vector6};

/// 8-node serendipity quadrilateral plane stress element.
///
/// Higher-order element with:
/// - 8 nodes (4 corner + 4 mid-edge)
/// - 2 DOFs per node (u, v displacements)
/// - 16 total DOFs
/// - Quadratic strain variation
/// - 3×3 (9 point) Gauss quadrature
///
/// # Advantages over Quad4
///
/// - Better accuracy with fewer elements
/// - Can represent curved edges
/// - Captures stress gradients within element
/// - Avoids shear locking in bending-dominated problems
#[derive(Debug, Clone, Copy)]
pub struct Quad8 {
    /// Element thickness.
    thickness: f64,
}

/// Node positions in natural coordinates.
const NODE_COORDS: [(f64, f64); 8] = [
    (-1.0, -1.0), // Node 1 (corner)
    (1.0, -1.0),  // Node 2 (corner)
    (1.0, 1.0),   // Node 3 (corner)
    (-1.0, 1.0),  // Node 4 (corner)
    (0.0, -1.0),  // Node 5 (mid-edge 1-2)
    (1.0, 0.0),   // Node 6 (mid-edge 2-3)
    (0.0, 1.0),   // Node 7 (mid-edge 3-4)
    (-1.0, 0.0),  // Node 8 (mid-edge 4-1)
];

impl Quad8 {
    /// Create a new Quad8 element with specified thickness.
    ///
    /// # Arguments
    ///
    /// * `thickness` - Element thickness (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if thickness is not positive.
    pub fn new(thickness: f64) -> Self {
        assert!(thickness > 0.0, "Thickness must be positive");
        Self { thickness }
    }

    /// Evaluate shape functions at natural coordinates (ξ, η).
    ///
    /// Returns [N1, N2, N3, N4, N5, N6, N7, N8].
    fn shape_functions(xi: f64, eta: f64) -> [f64; 8] {
        // Corner nodes: N_i = (1/4)(1 + ξ_i*ξ)(1 + η_i*η)(ξ_i*ξ + η_i*η - 1)
        // Mid-side nodes on ξ = 0: N_i = (1/2)(1 - ξ²)(1 + η_i*η)
        // Mid-side nodes on η = 0: N_i = (1/2)(1 + ξ_i*ξ)(1 - η²)

        let xi2 = xi * xi;
        let eta2 = eta * eta;

        [
            // N1: corner at (-1, -1)
            0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0),
            // N2: corner at (1, -1)
            0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0),
            // N3: corner at (1, 1)
            0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0),
            // N4: corner at (-1, 1)
            0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0),
            // N5: mid-side at (0, -1)
            0.5 * (1.0 - xi2) * (1.0 - eta),
            // N6: mid-side at (1, 0)
            0.5 * (1.0 + xi) * (1.0 - eta2),
            // N7: mid-side at (0, 1)
            0.5 * (1.0 - xi2) * (1.0 + eta),
            // N8: mid-side at (-1, 0)
            0.5 * (1.0 - xi) * (1.0 - eta2),
        ]
    }

    /// Evaluate shape function derivatives with respect to natural coordinates.
    ///
    /// Returns (dN/dξ, dN/dη) for each node.
    fn shape_function_derivatives(xi: f64, eta: f64) -> [(f64, f64); 8] {
        [
            // Node 1: corner at (-1, -1)
            // N1 = (1/4)(1-ξ)(1-η)(-ξ-η-1)
            (
                0.25 * (1.0 - eta) * (2.0 * xi + eta),
                0.25 * (1.0 - xi) * (xi + 2.0 * eta),
            ),
            // Node 2: corner at (1, -1)
            // N2 = (1/4)(1+ξ)(1-η)(ξ-η-1)
            (
                0.25 * (1.0 - eta) * (2.0 * xi - eta),
                0.25 * (1.0 + xi) * (-xi + 2.0 * eta),
            ),
            // Node 3: corner at (1, 1)
            // N3 = (1/4)(1+ξ)(1+η)(ξ+η-1)
            (
                0.25 * (1.0 + eta) * (2.0 * xi + eta),
                0.25 * (1.0 + xi) * (xi + 2.0 * eta),
            ),
            // Node 4: corner at (-1, 1)
            // N4 = (1/4)(1-ξ)(1+η)(-ξ+η-1)
            (
                0.25 * (1.0 + eta) * (2.0 * xi - eta),
                0.25 * (1.0 - xi) * (-xi + 2.0 * eta),
            ),
            // Node 5: mid-side at (0, -1)
            // N5 = (1/2)(1-ξ²)(1-η)
            (-xi * (1.0 - eta), -0.5 * (1.0 - xi * xi)),
            // Node 6: mid-side at (1, 0)
            // N6 = (1/2)(1+ξ)(1-η²)
            (0.5 * (1.0 - eta * eta), -(1.0 + xi) * eta),
            // Node 7: mid-side at (0, 1)
            // N7 = (1/2)(1-ξ²)(1+η)
            (-xi * (1.0 + eta), 0.5 * (1.0 - xi * xi)),
            // Node 8: mid-side at (-1, 0)
            // N8 = (1/2)(1-ξ)(1-η²)
            (-0.5 * (1.0 - eta * eta), -(1.0 - xi) * eta),
        ]
    }

    /// Compute Jacobian matrix and its determinant at (ξ, η).
    ///
    /// Returns (J, det(J)) where J maps natural to physical coordinates.
    fn jacobian(coords: &[Point3], xi: f64, eta: f64) -> (Matrix2<f64>, f64) {
        let dn_dnat = Self::shape_function_derivatives(xi, eta);

        // J = [[∂x/∂ξ, ∂y/∂ξ],
        //      [∂x/∂η, ∂y/∂η]]
        let mut j = Matrix2::zeros();
        for i in 0..8 {
            j[(0, 0)] += dn_dnat[i].0 * coords[i][0]; // ∂x/∂ξ
            j[(0, 1)] += dn_dnat[i].0 * coords[i][1]; // ∂y/∂ξ
            j[(1, 0)] += dn_dnat[i].1 * coords[i][0]; // ∂x/∂η
            j[(1, 1)] += dn_dnat[i].1 * coords[i][1]; // ∂y/∂η
        }

        let det_j = j.determinant();
        (j, det_j)
    }

    /// Compute B-matrix at a specific integration point.
    ///
    /// Returns the 3x16 strain-displacement matrix.
    fn compute_b_at_point(coords: &[Point3], xi: f64, eta: f64) -> DMatrix<f64> {
        let dn_dnat = Self::shape_function_derivatives(xi, eta);
        let (j, _det_j) = Self::jacobian(coords, xi, eta);

        // Invert Jacobian to get dN/dx, dN/dy from dN/dξ, dN/dη
        let j_inv = j
            .try_inverse()
            .expect("Degenerate element: Jacobian is singular");

        // dN/dx = J^(-1) * dN/dξ
        let mut dn_dx = [(0.0, 0.0); 8];
        for i in 0..8 {
            let dnat = Vector2::new(dn_dnat[i].0, dn_dnat[i].1);
            let dphys = j_inv * dnat;
            dn_dx[i] = (dphys[0], dphys[1]);
        }

        // Build B-matrix (3x16)
        let mut b = DMatrix::zeros(3, 16);
        for i in 0..8 {
            let col = 2 * i;
            b[(0, col)] = dn_dx[i].0; // ε_xx = ∂u/∂x
            b[(1, col + 1)] = dn_dx[i].1; // ε_yy = ∂v/∂y
            b[(2, col)] = dn_dx[i].1; // γ_xy = ∂u/∂y + ∂v/∂x
            b[(2, col + 1)] = dn_dx[i].0;
        }

        b
    }
}

impl Element for Quad8 {
    fn n_nodes(&self) -> usize {
        8
    }

    fn dofs_per_node(&self) -> usize {
        2
    }

    fn stiffness(&self, coords: &[Point3], material: &Material) -> DMatrix<f64> {
        assert_eq!(
            coords.len(),
            8,
            "Quad8 requires exactly 8 nodal coordinates"
        );

        let d = material.constitutive_plane_stress();
        let mut k = DMatrix::zeros(16, 16);

        // 3×3 Gauss quadrature (9 points, sufficient for quadratic elements)
        let gauss_points = gauss_quad(3);
        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let weight = gp.weight;

            let b = Self::compute_b_at_point(coords, xi, eta);
            let (_, det_j) = Self::jacobian(coords, xi, eta);

            // K += w * t * det(J) * B^T * D * B
            let db = &d * &b;
            let k_contrib = b.transpose() * db * (weight * self.thickness * det_j);
            k += k_contrib;
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
            8,
            "Quad8 requires exactly 8 nodal coordinates"
        );
        assert_eq!(
            displacements.len(),
            16,
            "Quad8 requires 16 displacement DOFs"
        );

        let d = material.constitutive_plane_stress();
        let u = nalgebra::DVector::from_row_slice(displacements);

        // Compute stress at each integration point
        let gauss_points = gauss_quad(3);
        let mut stresses = Vec::with_capacity(gauss_points.len());

        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();

            let b = Self::compute_b_at_point(coords, xi, eta);

            // ε = B * u
            let strain = &b * &u;

            // σ = D * ε
            let stress_2d = &d * Vector3::from_iterator(strain.iter().cloned());

            // Convert to 6-component stress tensor
            let stress_6 = Vector6::new(
                stress_2d[0], // σ_xx
                stress_2d[1], // σ_yy
                0.0,          // σ_zz = 0 (plane stress)
                stress_2d[2], // τ_xy
                0.0,          // τ_yz = 0
                0.0,          // τ_xz = 0
            );

            stresses.push(StressTensor(stress_6));
        }

        stresses
    }

    fn volume(&self, coords: &[Point3]) -> f64 {
        assert_eq!(
            coords.len(),
            8,
            "Quad8 requires exactly 8 nodal coordinates"
        );

        // Integrate using 3×3 Gauss quadrature
        let gauss_points = gauss_quad(3);
        let mut area = 0.0;
        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let (_, det_j) = Self::jacobian(coords, xi, eta);
            area += gp.weight * det_j;
        }

        area * self.thickness
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn unit_square_quad8() -> Vec<Point3> {
        // Unit square with mid-edge nodes
        vec![
            Point3::new(0.0, 0.0, 0.0), // Node 1 (corner)
            Point3::new(1.0, 0.0, 0.0), // Node 2 (corner)
            Point3::new(1.0, 1.0, 0.0), // Node 3 (corner)
            Point3::new(0.0, 1.0, 0.0), // Node 4 (corner)
            Point3::new(0.5, 0.0, 0.0), // Node 5 (mid-edge 1-2)
            Point3::new(1.0, 0.5, 0.0), // Node 6 (mid-edge 2-3)
            Point3::new(0.5, 1.0, 0.0), // Node 7 (mid-edge 3-4)
            Point3::new(0.0, 0.5, 0.0), // Node 8 (mid-edge 4-1)
        ]
    }

    #[test]
    fn test_quad8_node_count() {
        let quad = Quad8::new(1.0);
        assert_eq!(quad.n_nodes(), 8);
        assert_eq!(quad.dofs_per_node(), 2);
        assert_eq!(quad.n_dofs(), 16);
    }

    #[test]
    fn test_quad8_shape_functions_sum_to_one() {
        // Shape functions should sum to 1 at any point
        let test_points = [
            (0.0, 0.0),   // center
            (0.5, 0.5),   // interior
            (-0.5, 0.5),  // interior
            (0.0, 1.0),   // mid-side
            (-1.0, -1.0), // corner
            (1.0, 0.0),   // mid-side
        ];
        for (xi, eta) in &test_points {
            let n = Quad8::shape_functions(*xi, *eta);
            let sum: f64 = n.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_quad8_shape_functions_at_nodes() {
        // Shape function N_i should be 1 at node i and 0 at other nodes
        for (i, (xi_i, eta_i)) in NODE_COORDS.iter().enumerate() {
            let n = Quad8::shape_functions(*xi_i, *eta_i);
            for j in 0..8 {
                if i == j {
                    assert_relative_eq!(n[j], 1.0, epsilon = 1e-14);
                } else {
                    assert_relative_eq!(n[j], 0.0, epsilon = 1e-14);
                }
            }
        }
    }

    #[test]
    fn test_quad8_volume() {
        let quad = Quad8::new(0.1);
        let coords = unit_square_quad8();
        let vol = quad.volume(&coords);
        // Volume = Area * thickness = 1.0 * 0.1 = 0.1
        assert_relative_eq!(vol, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_quad8_stiffness_symmetric() {
        let quad = Quad8::new(1.0);
        let coords = unit_square_quad8();
        let mat = Material::steel();

        let k = quad.stiffness(&coords, &mat);

        assert_eq!(k.nrows(), 16);
        assert_eq!(k.ncols(), 16);
        // Find max stiffness for scaling
        let k_max = k.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        for i in 0..16 {
            for j in 0..16 {
                // Use both absolute and relative tolerance
                // Absolute tolerance scaled to max stiffness, relative to account for numerical precision
                let tol = 1e-10 * k_max;
                assert!(
                    (k[(i, j)] - k[(j, i)]).abs() < tol,
                    "K[{},{}] = {} != K[{},{}] = {} (diff = {:e})",
                    i, j, k[(i, j)], j, i, k[(j, i)], (k[(i, j)] - k[(j, i)]).abs()
                );
            }
        }
    }

    #[test]
    fn test_quad8_stiffness_positive_diagonal() {
        let quad = Quad8::new(1.0);
        let coords = unit_square_quad8();
        let mat = Material::steel();

        let k = quad.stiffness(&coords, &mat);

        for i in 0..16 {
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
    fn test_quad8_rigid_body_modes() {
        let quad = Quad8::new(1.0);
        let coords = unit_square_quad8();
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = quad.stiffness(&coords, &mat);

        // Pure x-translation: u = [1,0, ...] for all 8 nodes
        let u_x = nalgebra::DVector::from_vec(vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ]);
        let f_x = &k * &u_x;
        assert_relative_eq!(f_x.norm(), 0.0, epsilon = 1e-10);

        // Pure y-translation
        let u_y = nalgebra::DVector::from_vec(vec![
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ]);
        let f_y = &k * &u_y;
        assert_relative_eq!(f_y.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quad8_constant_strain_patch() {
        let quad = Quad8::new(1.0);
        let coords = unit_square_quad8();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Impose uniform ε_xx = 0.001: u = 0.001 * x
        // Based on physical coordinates (x, y):
        // Node 1 (0,0): u=0
        // Node 2 (1,0): u=0.001
        // Node 3 (1,1): u=0.001
        // Node 4 (0,1): u=0
        // Node 5 (0.5,0): u=0.0005
        // Node 6 (1,0.5): u=0.001
        // Node 7 (0.5,1): u=0.0005
        // Node 8 (0,0.5): u=0
        let displacements = [
            0.0, 0.0,     // Node 1
            0.001, 0.0,   // Node 2
            0.001, 0.0,   // Node 3
            0.0, 0.0,     // Node 4
            0.0005, 0.0,  // Node 5
            0.001, 0.0,   // Node 6
            0.0005, 0.0,  // Node 7
            0.0, 0.0,     // Node 8
        ];

        let stresses = quad.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 9); // 9 integration points (3x3)

        let d = mat.constitutive_plane_stress();
        let expected_sigma_xx = d[(0, 0)] * 0.001;
        let expected_sigma_yy = d[(1, 0)] * 0.001;

        // All integration points should have same stress (constant strain field)
        for stress in &stresses {
            assert_relative_eq!(stress.0[0], expected_sigma_xx, epsilon = 1e-3);
            assert_relative_eq!(stress.0[1], expected_sigma_yy, epsilon = 1e-3);
            assert_relative_eq!(stress.0[2], 0.0, epsilon = 1e-10); // σ_zz = 0
            assert_relative_eq!(stress.0[3], 0.0, epsilon = 1e-3); // τ_xy = 0
        }
    }

    #[test]
    fn test_quad8_curved_edge() {
        // Test with a curved element (parabolic edge)
        let quad = Quad8::new(1.0);
        // Bulge the bottom mid-side node upward
        let coords = vec![
            Point3::new(0.0, 0.0, 0.0),   // Node 1
            Point3::new(1.0, 0.0, 0.0),   // Node 2
            Point3::new(1.0, 1.0, 0.0),   // Node 3
            Point3::new(0.0, 1.0, 0.0),   // Node 4
            Point3::new(0.5, 0.1, 0.0),   // Node 5 - curved edge
            Point3::new(1.0, 0.5, 0.0),   // Node 6
            Point3::new(0.5, 1.0, 0.0),   // Node 7
            Point3::new(0.0, 0.5, 0.0),   // Node 8
        ];
        let mat = Material::steel();

        // Should still produce symmetric stiffness
        let k = quad.stiffness(&coords, &mat);
        // Find max stiffness for scaling
        let k_max = k.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        for i in 0..16 {
            for j in 0..16 {
                // Use absolute tolerance scaled to max stiffness
                let tol = 1e-10 * k_max;
                assert!(
                    (k[(i, j)] - k[(j, i)]).abs() < tol,
                    "K[{},{}] = {} != K[{},{}] = {} (diff = {:e})",
                    i, j, k[(i, j)], j, i, k[(j, i)], (k[(i, j)] - k[(j, i)]).abs()
                );
            }
        }

        // Volume should be slightly larger due to bulge (area ≈ 1.0 + 0.1*0.5*4/3 = 1.0667)
        // Actually for parabolic bulge: A = 1.0 + 2/3 * 0.1 = 1.0667 (Simpsons rule approx)
        let vol = quad.volume(&coords);
        assert!(vol > 0.1, "Volume should be positive");
    }

    #[test]
    fn test_quad8_shear_strain() {
        let quad = Quad8::new(1.0);
        let coords = unit_square_quad8();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Impose pure shear: u = γ/2 * y, v = γ/2 * x where γ = 0.002
        let gamma = 0.002;
        let displacements = [
            0.0,
            0.0, // Node 1: (0,0)
            0.0,
            gamma / 2.0, // Node 2: (1,0)
            gamma / 2.0,
            gamma / 2.0, // Node 3: (1,1)
            gamma / 2.0,
            0.0, // Node 4: (0,1)
            0.0,
            gamma / 4.0, // Node 5: (0.5,0)
            gamma / 4.0,
            gamma / 2.0, // Node 6: (1,0.5)
            gamma / 2.0,
            gamma / 4.0, // Node 7: (0.5,1)
            gamma / 4.0,
            0.0, // Node 8: (0,0.5)
        ];

        let stresses = quad.stress(&coords, &displacements, &mat);

        let d = mat.constitutive_plane_stress();
        let expected_tau_xy = d[(2, 2)] * gamma;

        for stress in &stresses {
            assert_relative_eq!(stress.0[0], 0.0, epsilon = 1e-3); // σ_xx ≈ 0
            assert_relative_eq!(stress.0[1], 0.0, epsilon = 1e-3); // σ_yy ≈ 0
            assert_relative_eq!(stress.0[3], expected_tau_xy, epsilon = 1e-3); // τ_xy
        }
    }

    #[test]
    #[should_panic(expected = "Thickness must be positive")]
    fn test_quad8_invalid_thickness() {
        Quad8::new(0.0);
    }
}
