//! 6-node quadratic triangular plane stress element (Tri6).
//!
//! The Tri6 element is a higher-order 2D element with quadratic shape functions:
//! - 6 nodes: 3 corner nodes + 3 mid-edge nodes
//! - 2 DOFs per node (u, v displacements)
//! - 12 total DOFs
//! - Quadratic strain variation within element
//! - 3-point Gauss quadrature for integration
//!
//! # Node Numbering
//!
//! ```text
//!        3
//!       /\
//!      /  \
//!     6    5
//!    /      \
//!   /        \
//!  1----4-----2
//! ```
//!
//! - Corner nodes: 1, 2, 3 (local indices 0, 1, 2)
//! - Mid-edge nodes: 4 (edge 1-2), 5 (edge 2-3), 6 (edge 3-1) (local indices 3, 4, 5)
//!
//! # Natural Coordinates
//!
//! Uses area (barycentric) coordinates (L1, L2, L3) where:
//! - L1 + L2 + L3 = 1
//! - L_i = A_i / A (ratio of sub-triangle area to total area)
//! - Corner nodes: L_i = 1 at node i, 0 at others
//! - Mid-edge nodes: L_i = 0.5 for both adjacent corners
//!
//! # Shape Functions
//!
//! ```text
//! N1 = L1 * (2*L1 - 1)   (corner 1)
//! N2 = L2 * (2*L2 - 1)   (corner 2)
//! N3 = L3 * (2*L3 - 1)   (corner 3)
//! N4 = 4 * L1 * L2       (mid-edge 1-2)
//! N5 = 4 * L2 * L3       (mid-edge 2-3)
//! N6 = 4 * L3 * L1       (mid-edge 3-1)
//! ```

use crate::element::gauss::gauss_tri;
use crate::element::Element;
use crate::material::Material;
use crate::types::{Point3, StressTensor};
use nalgebra::{DMatrix, Matrix2, Vector2, Vector3, Vector6};

/// 6-node quadratic triangular plane stress element.
///
/// Higher-order element with:
/// - 6 nodes (3 corner + 3 mid-edge)
/// - 2 DOFs per node (u, v displacements)
/// - 12 total DOFs
/// - Quadratic strain variation
/// - 3-point Gauss quadrature
///
/// # Advantages over Tri3
///
/// - Better accuracy with fewer elements
/// - Can represent curved edges
/// - Captures stress gradients within element
/// - Avoids volumetric locking better than linear elements
#[derive(Debug, Clone, Copy)]
pub struct Tri6 {
    /// Element thickness.
    thickness: f64,
}

impl Tri6 {
    /// Create a new Tri6 element with specified thickness.
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

    /// Evaluate shape functions at area coordinates (L1, L2, L3).
    ///
    /// Returns [N1, N2, N3, N4, N5, N6].
    fn shape_functions(l1: f64, l2: f64, l3: f64) -> [f64; 6] {
        [
            l1 * (2.0 * l1 - 1.0), // N1: corner 1
            l2 * (2.0 * l2 - 1.0), // N2: corner 2
            l3 * (2.0 * l3 - 1.0), // N3: corner 3
            4.0 * l1 * l2,         // N4: mid-edge 1-2
            4.0 * l2 * l3,         // N5: mid-edge 2-3
            4.0 * l3 * l1,         // N6: mid-edge 3-1
        ]
    }

    /// Evaluate shape function derivatives with respect to area coordinates.
    ///
    /// Returns derivatives dN/dL1, dN/dL2, dN/dL3 for each shape function.
    /// Each tuple is (dN_i/dL1, dN_i/dL2, dN_i/dL3).
    fn shape_function_derivatives_area(l1: f64, l2: f64, l3: f64) -> [(f64, f64, f64); 6] {
        [
            // dN1/dL1, dN1/dL2, dN1/dL3
            (4.0 * l1 - 1.0, 0.0, 0.0),
            // dN2/dL1, dN2/dL2, dN2/dL3
            (0.0, 4.0 * l2 - 1.0, 0.0),
            // dN3/dL1, dN3/dL2, dN3/dL3
            (0.0, 0.0, 4.0 * l3 - 1.0),
            // dN4/dL1, dN4/dL2, dN4/dL3
            (4.0 * l2, 4.0 * l1, 0.0),
            // dN5/dL1, dN5/dL2, dN5/dL3
            (0.0, 4.0 * l3, 4.0 * l2),
            // dN6/dL1, dN6/dL2, dN6/dL3
            (4.0 * l3, 0.0, 4.0 * l1),
        ]
    }

    /// Convert area coordinate derivatives to physical coordinate derivatives.
    ///
    /// Given dN/dL_i and nodal coordinates, compute dN/dx and dN/dy.
    ///
    /// The mapping uses parametric coordinates (ξ, η) where:
    /// - L1 = 1 - ξ - η
    /// - L2 = ξ
    /// - L3 = η
    ///
    /// So the parametric derivatives are:
    /// - dN/dξ = dN/dL2 - dN/dL1 (since ∂L2/∂ξ = 1, ∂L1/∂ξ = -1)
    /// - dN/dη = dN/dL3 - dN/dL1 (since ∂L3/∂η = 1, ∂L1/∂η = -1)
    ///
    /// The Jacobian J maps:
    /// [dx, dy]^T = J * [dξ, dη]^T
    ///
    /// And we compute dN/dx, dN/dy via: [dN/dx, dN/dy]^T = J^(-1) * [dN/dξ, dN/dη]^T
    ///
    /// Returns (physical derivatives, |det(J)|) where det(J) is always positive.
    fn compute_physical_derivatives(
        coords: &[Point3],
        l1: f64,
        l2: f64,
        l3: f64,
    ) -> ([(f64, f64); 6], f64) {
        let dn_dl = Self::shape_function_derivatives_area(l1, l2, l3);

        // Convert from barycentric to parametric derivatives:
        // dN/dξ = dN/dL2 - dN/dL1
        // dN/dη = dN/dL3 - dN/dL1
        let mut dn_dxi = [0.0; 6];
        let mut dn_deta = [0.0; 6];
        for i in 0..6 {
            dn_dxi[i] = dn_dl[i].1 - dn_dl[i].0;  // dN/dL2 - dN/dL1
            dn_deta[i] = dn_dl[i].2 - dn_dl[i].0; // dN/dL3 - dN/dL1
        }

        // Compute Jacobian: J = [dx/dξ  dy/dξ]
        //                        [dx/dη  dy/dη]
        // where dx/dξ = sum_i (dN_i/dξ) * x_i
        let mut j = Matrix2::zeros();
        for i in 0..6 {
            j[(0, 0)] += dn_dxi[i] * coords[i][0];  // dx/dξ
            j[(0, 1)] += dn_dxi[i] * coords[i][1];  // dy/dξ
            j[(1, 0)] += dn_deta[i] * coords[i][0]; // dx/dη
            j[(1, 1)] += dn_deta[i] * coords[i][1]; // dy/dη
        }

        let det_j = j.determinant();

        // For a properly oriented element, det_j should be positive.
        // We take the absolute value for the integration weight.
        let j_inv = j
            .try_inverse()
            .expect("Degenerate element: Jacobian is singular");

        // Convert parametric derivatives to physical derivatives:
        // [dN/dx, dN/dy]^T = J^(-1) * [dN/dξ, dN/dη]^T
        let mut dn_dphys = [(0.0, 0.0); 6];
        for i in 0..6 {
            let dn_dnat = Vector2::new(dn_dxi[i], dn_deta[i]);
            let dn_dxy = j_inv * dn_dnat;
            dn_dphys[i] = (dn_dxy[0], dn_dxy[1]);
        }

        // Return absolute value of determinant for integration weight
        (dn_dphys, det_j.abs())
    }

    /// Compute B-matrix at a specific integration point.
    ///
    /// Returns the 3x12 strain-displacement matrix.
    fn compute_b_at_point(coords: &[Point3], l1: f64, l2: f64, l3: f64) -> (DMatrix<f64>, f64) {
        let (dn_dphys, det_j) = Self::compute_physical_derivatives(coords, l1, l2, l3);

        // Build B-matrix (3x12)
        let mut b = DMatrix::zeros(3, 12);
        for i in 0..6 {
            let col = 2 * i;
            b[(0, col)] = dn_dphys[i].0; // ε_xx = ∂u/∂x
            b[(1, col + 1)] = dn_dphys[i].1; // ε_yy = ∂v/∂y
            b[(2, col)] = dn_dphys[i].1; // γ_xy = ∂u/∂y + ∂v/∂x
            b[(2, col + 1)] = dn_dphys[i].0;
        }

        (b, det_j)
    }
}

impl Element for Tri6 {
    fn n_nodes(&self) -> usize {
        6
    }

    fn dofs_per_node(&self) -> usize {
        2
    }

    fn stiffness(&self, coords: &[Point3], material: &Material) -> DMatrix<f64> {
        assert_eq!(
            coords.len(),
            6,
            "Tri6 requires exactly 6 nodal coordinates"
        );

        let d = material.constitutive_plane_stress();
        let mut k = DMatrix::zeros(12, 12);

        // 3-point Gauss quadrature for triangles (degree 2, sufficient for quadratic)
        // Note: gauss_tri weights are scaled for unit triangle (area = 1/2), so we need
        // to multiply by 2 to get the reference triangle integration, then by det_j
        // for the physical element. However, det_j here is |J| which already gives us
        // the area ratio, so: K = w * 2 * t * det_j * B^T * D * B
        let gauss_points = gauss_tri(3);
        for gp in &gauss_points {
            let l1 = gp.coords[0];
            let l2 = gp.coords[1];
            let l3 = gp.coords[2];
            let weight = gp.weight;

            let (b, det_j) = Self::compute_b_at_point(coords, l1, l2, l3);

            // K += w * t * det(J) * B^T * D * B
            // The gauss_tri weights are scaled for unit triangle integration
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
            6,
            "Tri6 requires exactly 6 nodal coordinates"
        );
        assert_eq!(
            displacements.len(),
            12,
            "Tri6 requires 12 displacement DOFs"
        );

        let d = material.constitutive_plane_stress();
        let u = nalgebra::DVector::from_row_slice(displacements);

        // Compute stress at each integration point
        let gauss_points = gauss_tri(3);
        let mut stresses = Vec::with_capacity(gauss_points.len());

        for gp in &gauss_points {
            let l1 = gp.coords[0];
            let l2 = gp.coords[1];
            let l3 = gp.coords[2];

            let (b, _) = Self::compute_b_at_point(coords, l1, l2, l3);

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
            6,
            "Tri6 requires exactly 6 nodal coordinates"
        );

        // Integrate using 3-point Gauss quadrature
        // gauss_tri weights are scaled for unit triangle (sum to 0.5)
        let gauss_points = gauss_tri(3);
        let mut area = 0.0;
        for gp in &gauss_points {
            let l1 = gp.coords[0];
            let l2 = gp.coords[1];
            let l3 = gp.coords[2];
            let (_, det_j) = Self::compute_physical_derivatives(coords, l1, l2, l3);
            area += gp.weight * det_j;
        }

        area * self.thickness
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn unit_right_triangle_tri6() -> Vec<Point3> {
        // Unit right triangle with mid-edge nodes
        // Corner nodes
        vec![
            Point3::new(0.0, 0.0, 0.0), // Node 1 (corner)
            Point3::new(1.0, 0.0, 0.0), // Node 2 (corner)
            Point3::new(0.0, 1.0, 0.0), // Node 3 (corner)
            Point3::new(0.5, 0.0, 0.0), // Node 4 (mid-edge 1-2)
            Point3::new(0.5, 0.5, 0.0), // Node 5 (mid-edge 2-3)
            Point3::new(0.0, 0.5, 0.0), // Node 6 (mid-edge 3-1)
        ]
    }

    #[test]
    fn test_tri6_node_count() {
        let tri = Tri6::new(1.0);
        assert_eq!(tri.n_nodes(), 6);
        assert_eq!(tri.dofs_per_node(), 2);
        assert_eq!(tri.n_dofs(), 12);
    }

    #[test]
    fn test_tri6_shape_functions_sum_to_one() {
        // Shape functions should sum to 1 at any point
        let test_points = [
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), // centroid
            (0.5, 0.5, 0.0),                   // edge midpoint
            (0.25, 0.25, 0.5),                 // interior point
            (1.0, 0.0, 0.0),                   // corner 1
            (0.0, 1.0, 0.0),                   // corner 2
        ];
        for (l1, l2, l3) in &test_points {
            let n = Tri6::shape_functions(*l1, *l2, *l3);
            let sum: f64 = n.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_tri6_shape_functions_at_corner_nodes() {
        // At corner 1 (L1=1, L2=0, L3=0): N1=1, all others=0
        let n = Tri6::shape_functions(1.0, 0.0, 0.0);
        assert_relative_eq!(n[0], 1.0, epsilon = 1e-14);
        for i in 1..6 {
            assert_relative_eq!(n[i], 0.0, epsilon = 1e-14);
        }

        // At corner 2 (L1=0, L2=1, L3=0): N2=1
        let n = Tri6::shape_functions(0.0, 1.0, 0.0);
        assert_relative_eq!(n[1], 1.0, epsilon = 1e-14);

        // At corner 3 (L1=0, L2=0, L3=1): N3=1
        let n = Tri6::shape_functions(0.0, 0.0, 1.0);
        assert_relative_eq!(n[2], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_tri6_shape_functions_at_midedge_nodes() {
        // At mid-edge 1-2 (L1=0.5, L2=0.5, L3=0): N4=1
        let n = Tri6::shape_functions(0.5, 0.5, 0.0);
        assert_relative_eq!(n[3], 1.0, epsilon = 1e-14);
        assert_relative_eq!(n[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(n[1], 0.0, epsilon = 1e-14);

        // At mid-edge 2-3 (L1=0, L2=0.5, L3=0.5): N5=1
        let n = Tri6::shape_functions(0.0, 0.5, 0.5);
        assert_relative_eq!(n[4], 1.0, epsilon = 1e-14);

        // At mid-edge 3-1 (L1=0.5, L2=0, L3=0.5): N6=1
        let n = Tri6::shape_functions(0.5, 0.0, 0.5);
        assert_relative_eq!(n[5], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_tri6_volume() {
        let tri = Tri6::new(0.1);
        let coords = unit_right_triangle_tri6();
        let vol = tri.volume(&coords);
        // Volume = Area * thickness = 0.5 * 0.1 = 0.05
        assert_relative_eq!(vol, 0.05, epsilon = 1e-10);
    }

    #[test]
    fn test_tri6_stiffness_symmetric() {
        let tri = Tri6::new(1.0);
        let coords = unit_right_triangle_tri6();
        let mat = Material::steel();

        let k = tri.stiffness(&coords, &mat);

        assert_eq!(k.nrows(), 12);
        assert_eq!(k.ncols(), 12);
        for i in 0..12 {
            for j in 0..12 {
                assert_relative_eq!(k[(i, j)], k[(j, i)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_tri6_stiffness_positive_diagonal() {
        let tri = Tri6::new(1.0);
        let coords = unit_right_triangle_tri6();
        let mat = Material::steel();

        let k = tri.stiffness(&coords, &mat);

        for i in 0..12 {
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
    fn test_tri6_rigid_body_modes() {
        let tri = Tri6::new(1.0);
        let coords = unit_right_triangle_tri6();
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = tri.stiffness(&coords, &mat);

        // Pure x-translation: u = [1,0, 1,0, 1,0, 1,0, 1,0, 1,0]
        let u_x =
            nalgebra::DVector::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let f_x = &k * &u_x;
        assert_relative_eq!(f_x.norm(), 0.0, epsilon = 1e-10);

        // Pure y-translation
        let u_y =
            nalgebra::DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let f_y = &k * &u_y;
        assert_relative_eq!(f_y.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tri6_constant_strain_patch() {
        let tri = Tri6::new(1.0);
        let coords = unit_right_triangle_tri6();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Impose uniform ε_xx = 0.001: u = 0.001 * x
        // Node 1 (0,0): u=0
        // Node 2 (1,0): u=0.001
        // Node 3 (0,1): u=0
        // Node 4 (0.5,0): u=0.0005
        // Node 5 (0.5,0.5): u=0.0005
        // Node 6 (0,0.5): u=0
        let displacements = [
            0.0, 0.0,     // Node 1
            0.001, 0.0,   // Node 2
            0.0, 0.0,     // Node 3
            0.0005, 0.0,  // Node 4
            0.0005, 0.0,  // Node 5
            0.0, 0.0,     // Node 6
        ];

        let stresses = tri.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 3); // 3 integration points

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
    #[should_panic(expected = "Thickness must be positive")]
    fn test_tri6_invalid_thickness() {
        Tri6::new(0.0);
    }
}
