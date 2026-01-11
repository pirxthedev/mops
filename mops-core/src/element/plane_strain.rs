//! Plane strain elements for 2D analysis.
//!
//! Plane strain elements are used for thick structures where out-of-plane
//! strains are negligible (ε_z = 0). Common applications include dams, tunnels,
//! retaining walls, and long prismatic structures under transverse loading.
//!
//! This module provides:
//! - [`Tri3PlaneStrain`] - 3-node triangle (Constant Strain Triangle, CST)
//! - [`Quad4PlaneStrain`] - 4-node quadrilateral (bilinear)
//!
//! # Coordinate System
//!
//! For 2D elements, nodes have (x, y) coordinates but are stored as Point3
//! with z=0. Each node has 2 DOFs: (u, v) displacements.
//!
//! # Thickness Parameter
//!
//! The element thickness is specified at construction time. For plane strain,
//! this typically represents a unit depth (thickness = 1.0) for per-unit-depth
//! analysis.
//!
//! # Stress Output
//!
//! Stresses are returned as 6-component tensors for compatibility with 3D
//! elements. For plane strain: ε_z = γ_yz = γ_xz = 0, but σ_z ≠ 0 (computed from
//! σ_z = ν(σ_x + σ_y)).

use crate::element::gauss::gauss_quad;
use crate::element::Element;
use crate::material::Material;
use crate::types::{Point3, StressTensor};
use nalgebra::{DMatrix, Matrix2, Vector2, Vector3, Vector6};

/// 3-node triangular plane strain element (Constant Strain Triangle).
///
/// The simplest 2D element with:
/// - 3 nodes at vertices
/// - 2 DOFs per node (u, v displacements)
/// - 6 total DOFs
/// - Constant strain/stress within element
/// - Single integration point at centroid
///
/// # Plane Strain Assumption
///
/// ε_z = γ_yz = γ_xz = 0, meaning the structure is constrained in the z-direction.
/// This leads to non-zero σ_z = ν(σ_x + σ_y).
///
/// # Limitations
///
/// - Volumetric locking in nearly incompressible materials
/// - Low accuracy - requires fine meshes
/// - Poor performance in bending-dominated problems
#[derive(Debug, Clone, Copy)]
pub struct Tri3PlaneStrain {
    /// Element thickness (typically 1.0 for per-unit-depth analysis).
    thickness: f64,
}

impl Tri3PlaneStrain {
    /// Create a new Tri3PlaneStrain element with specified thickness.
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

    /// Compute area of triangle from nodal coordinates.
    fn compute_area(coords: &[Point3]) -> f64 {
        assert_eq!(coords.len(), 3, "Tri3 requires exactly 3 nodal coordinates");

        let x1 = coords[0][0];
        let y1 = coords[0][1];
        let x2 = coords[1][0];
        let y2 = coords[1][1];
        let x3 = coords[2][0];
        let y3 = coords[2][1];

        // Area = 0.5 * |det([x2-x1, x3-x1; y2-y1, y3-y1])|
        // = 0.5 * |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
        0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs()
    }

    /// Compute the B-matrix (strain-displacement) for constant strain.
    ///
    /// Returns the 3x6 B-matrix where ε = B * u
    /// ε = [ε_xx, ε_yy, γ_xy]^T
    ///
    /// Also returns the element area.
    fn compute_b_matrix(coords: &[Point3]) -> (DMatrix<f64>, f64) {
        assert_eq!(coords.len(), 3, "Tri3 requires exactly 3 nodal coordinates");

        let x1 = coords[0][0];
        let y1 = coords[0][1];
        let x2 = coords[1][0];
        let y2 = coords[1][1];
        let x3 = coords[2][0];
        let y3 = coords[2][1];

        // Area (with sign for orientation)
        let area_2_signed = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
        let area = area_2_signed.abs() / 2.0;

        // Shape function derivatives (constant for linear triangle):
        // dN1/dx = (y2 - y3) / (2*A)
        // dN1/dy = (x3 - x2) / (2*A)
        // dN2/dx = (y3 - y1) / (2*A)
        // dN2/dy = (x1 - x3) / (2*A)
        // dN3/dx = (y1 - y2) / (2*A)
        // dN3/dy = (x2 - x1) / (2*A)
        let inv_2a = 1.0 / area_2_signed; // Note: using signed area preserves orientation

        let dn1_dx = (y2 - y3) * inv_2a;
        let dn1_dy = (x3 - x2) * inv_2a;
        let dn2_dx = (y3 - y1) * inv_2a;
        let dn2_dy = (x1 - x3) * inv_2a;
        let dn3_dx = (y1 - y2) * inv_2a;
        let dn3_dy = (x2 - x1) * inv_2a;

        // B-matrix (3x6)
        // ε = [ε_xx, ε_yy, γ_xy]^T = B * [u1, v1, u2, v2, u3, v3]^T
        //
        // ε_xx = ∂u/∂x = dN1/dx * u1 + dN2/dx * u2 + dN3/dx * u3
        // ε_yy = ∂v/∂y = dN1/dy * v1 + dN2/dy * v2 + dN3/dy * v3
        // γ_xy = ∂u/∂y + ∂v/∂x
        let mut b = DMatrix::zeros(3, 6);

        // Node 1 (columns 0, 1)
        b[(0, 0)] = dn1_dx;
        b[(1, 1)] = dn1_dy;
        b[(2, 0)] = dn1_dy;
        b[(2, 1)] = dn1_dx;

        // Node 2 (columns 2, 3)
        b[(0, 2)] = dn2_dx;
        b[(1, 3)] = dn2_dy;
        b[(2, 2)] = dn2_dy;
        b[(2, 3)] = dn2_dx;

        // Node 3 (columns 4, 5)
        b[(0, 4)] = dn3_dx;
        b[(1, 5)] = dn3_dy;
        b[(2, 4)] = dn3_dy;
        b[(2, 5)] = dn3_dx;

        (b, area)
    }
}

impl Element for Tri3PlaneStrain {
    fn n_nodes(&self) -> usize {
        3
    }

    fn dofs_per_node(&self) -> usize {
        2
    }

    fn stiffness(&self, coords: &[Point3], material: &Material) -> DMatrix<f64> {
        let (b, area) = Self::compute_b_matrix(coords);

        // D is the 3x3 plane strain constitutive matrix
        let d = material.constitutive_plane_strain();

        // K = t * A * B^T * D * B (6x6)
        let db = &d * &b; // 3x6
        let k = b.transpose() * db * self.thickness * area; // 6x6

        k
    }

    fn stress(
        &self,
        coords: &[Point3],
        displacements: &[f64],
        material: &Material,
    ) -> Vec<StressTensor> {
        assert_eq!(
            displacements.len(),
            6,
            "Tri3 requires 6 displacement DOFs"
        );

        let (b, _area) = Self::compute_b_matrix(coords);
        let d = material.constitutive_plane_strain();

        // ε = B * u (3x1)
        let u = nalgebra::DVector::from_row_slice(displacements);
        let strain = &b * &u;

        // σ = D * ε (3x1) for in-plane stresses
        let stress_2d = &d * Vector3::from_iterator(strain.iter().cloned());

        // For plane strain, σ_z = ν(σ_x + σ_y)
        let nu = material.poissons_ratio;
        let sigma_z = nu * (stress_2d[0] + stress_2d[1]);

        // Convert to 6-component stress tensor
        let stress_6 = Vector6::new(
            stress_2d[0], // σ_xx
            stress_2d[1], // σ_yy
            sigma_z,      // σ_zz = ν(σ_x + σ_y) for plane strain
            stress_2d[2], // τ_xy
            0.0,          // τ_yz = 0
            0.0,          // τ_xz = 0
        );

        // Single stress tensor (constant strain element)
        vec![StressTensor(stress_6)]
    }

    fn volume(&self, coords: &[Point3]) -> f64 {
        Self::compute_area(coords) * self.thickness
    }
}

/// 4-node quadrilateral plane strain element.
///
/// A 2D element with:
/// - 4 nodes at corners
/// - 2 DOFs per node (u, v displacements)
/// - 8 total DOFs
/// - Bilinear shape functions
/// - 2×2 Gauss quadrature (4 integration points)
///
/// # Plane Strain Assumption
///
/// ε_z = γ_yz = γ_xz = 0, meaning the structure is constrained in the z-direction.
/// This leads to non-zero σ_z = ν(σ_x + σ_y).
///
/// # Shape Functions
///
/// In natural coordinates (ξ, η) ∈ [-1, 1]²:
/// ```text
/// N_i = (1 + ξ_i*ξ)(1 + η_i*η) / 4
/// ```
/// where (ξ_i, η_i) = (±1, ±1) for node i.
///
/// # Node Numbering
///
/// ```text
/// 4 --- 3
/// |     |
/// 1 --- 2
/// ```
/// Node 1: (-1, -1), Node 2: (+1, -1), Node 3: (+1, +1), Node 4: (-1, +1)
#[derive(Debug, Clone, Copy)]
pub struct Quad4PlaneStrain {
    /// Element thickness (typically 1.0 for per-unit-depth analysis).
    thickness: f64,
}

impl Quad4PlaneStrain {
    /// Create a new Quad4PlaneStrain element with specified thickness.
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

    /// Node positions in natural coordinates.
    const NODE_COORDS: [(f64, f64); 4] = [
        (-1.0, -1.0), // Node 1
        (1.0, -1.0),  // Node 2
        (1.0, 1.0),   // Node 3
        (-1.0, 1.0),  // Node 4
    ];

    /// Evaluate shape functions at natural coordinates (ξ, η).
    fn shape_functions(xi: f64, eta: f64) -> [f64; 4] {
        [
            0.25 * (1.0 - xi) * (1.0 - eta), // N1
            0.25 * (1.0 + xi) * (1.0 - eta), // N2
            0.25 * (1.0 + xi) * (1.0 + eta), // N3
            0.25 * (1.0 - xi) * (1.0 + eta), // N4
        ]
    }

    /// Evaluate shape function derivatives with respect to natural coordinates.
    ///
    /// Returns (dN/dξ, dN/dη) for each node.
    fn shape_function_derivatives(xi: f64, eta: f64) -> [(f64, f64); 4] {
        [
            (
                -0.25 * (1.0 - eta), // dN1/dξ
                -0.25 * (1.0 - xi),  // dN1/dη
            ),
            (
                0.25 * (1.0 - eta), // dN2/dξ
                -0.25 * (1.0 + xi), // dN2/dη
            ),
            (
                0.25 * (1.0 + eta), // dN3/dξ
                0.25 * (1.0 + xi),  // dN3/dη
            ),
            (
                -0.25 * (1.0 + eta), // dN4/dξ
                0.25 * (1.0 - xi),   // dN4/dη
            ),
        ]
    }

    /// Compute Jacobian matrix and its determinant at (ξ, η).
    ///
    /// Returns (J, det(J)) where J maps natural to physical coordinates.
    fn jacobian(coords: &[Point3], xi: f64, eta: f64) -> (Matrix2<f64>, f64) {
        let dn_dnat = Self::shape_function_derivatives(xi, eta);

        // J = [[∂x/∂ξ, ∂y/∂ξ],
        //      [∂x/∂η, ∂y/∂η]]
        // where ∂x/∂ξ = Σ dN_i/dξ * x_i
        let mut j = Matrix2::zeros();
        for i in 0..4 {
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
    /// Returns the 3x8 strain-displacement matrix.
    fn compute_b_at_point(coords: &[Point3], xi: f64, eta: f64) -> DMatrix<f64> {
        let dn_dnat = Self::shape_function_derivatives(xi, eta);
        let (j, _det_j) = Self::jacobian(coords, xi, eta);

        // Invert Jacobian to get dN/dx, dN/dy from dN/dξ, dN/dη
        let j_inv = j.try_inverse().expect("Degenerate element: Jacobian is singular");

        // dN/dx = J^(-1) * dN/dξ
        // [dN/dx]   [j11 j12] [dN/dξ]
        // [dN/dy] = [j21 j22] [dN/dη]
        let mut dn_dx = [(0.0, 0.0); 4];
        for i in 0..4 {
            let dnat = Vector2::new(dn_dnat[i].0, dn_dnat[i].1);
            let dphys = j_inv * dnat;
            dn_dx[i] = (dphys[0], dphys[1]);
        }

        // Build B-matrix (3x8)
        let mut b = DMatrix::zeros(3, 8);
        for i in 0..4 {
            let col = 2 * i;
            b[(0, col)] = dn_dx[i].0;     // ε_xx = ∂u/∂x
            b[(1, col + 1)] = dn_dx[i].1; // ε_yy = ∂v/∂y
            b[(2, col)] = dn_dx[i].1;     // γ_xy = ∂u/∂y + ∂v/∂x
            b[(2, col + 1)] = dn_dx[i].0;
        }

        b
    }
}

impl Element for Quad4PlaneStrain {
    fn n_nodes(&self) -> usize {
        4
    }

    fn dofs_per_node(&self) -> usize {
        2
    }

    fn stiffness(&self, coords: &[Point3], material: &Material) -> DMatrix<f64> {
        assert_eq!(
            coords.len(),
            4,
            "Quad4 requires exactly 4 nodal coordinates"
        );

        let d = material.constitutive_plane_strain();
        let mut k = DMatrix::zeros(8, 8);

        // 2x2 Gauss quadrature
        let gauss_points = gauss_quad(2);
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
            4,
            "Quad4 requires exactly 4 nodal coordinates"
        );
        assert_eq!(
            displacements.len(),
            8,
            "Quad4 requires 8 displacement DOFs"
        );

        let d = material.constitutive_plane_strain();
        let u = nalgebra::DVector::from_row_slice(displacements);
        let nu = material.poissons_ratio;

        // Compute stress at each integration point
        let gauss_points = gauss_quad(2);
        let mut stresses = Vec::with_capacity(gauss_points.len());

        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();

            let b = Self::compute_b_at_point(coords, xi, eta);

            // ε = B * u
            let strain = &b * &u;

            // σ = D * ε (in-plane stresses)
            let stress_2d = &d * Vector3::from_iterator(strain.iter().cloned());

            // For plane strain, σ_z = ν(σ_x + σ_y)
            let sigma_z = nu * (stress_2d[0] + stress_2d[1]);

            // Convert to 6-component stress tensor
            let stress_6 = Vector6::new(
                stress_2d[0], // σ_xx
                stress_2d[1], // σ_yy
                sigma_z,      // σ_zz = ν(σ_x + σ_y) for plane strain
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
            4,
            "Quad4 requires exactly 4 nodal coordinates"
        );

        // Integrate using 2x2 Gauss quadrature
        let gauss_points = gauss_quad(2);
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

    fn unit_right_triangle() -> Vec<Point3> {
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]
    }

    fn unit_square_quad() -> Vec<Point3> {
        // Node numbering: counter-clockwise from bottom-left
        vec![
            Point3::new(0.0, 0.0, 0.0), // Node 1
            Point3::new(1.0, 0.0, 0.0), // Node 2
            Point3::new(1.0, 1.0, 0.0), // Node 3
            Point3::new(0.0, 1.0, 0.0), // Node 4
        ]
    }

    // === Tri3PlaneStrain Tests ===

    #[test]
    fn test_tri3_plane_strain_area_unit() {
        let coords = unit_right_triangle();
        let area = Tri3PlaneStrain::compute_area(&coords);
        assert_relative_eq!(area, 0.5, epsilon = 1e-14);
    }

    #[test]
    fn test_tri3_plane_strain_volume() {
        let tri = Tri3PlaneStrain::new(0.1);
        let coords = unit_right_triangle();
        let vol = tri.volume(&coords);
        // Volume = Area * thickness = 0.5 * 0.1 = 0.05
        assert_relative_eq!(vol, 0.05, epsilon = 1e-14);
    }

    #[test]
    fn test_tri3_plane_strain_node_count() {
        let tri = Tri3PlaneStrain::new(1.0);
        assert_eq!(tri.n_nodes(), 3);
        assert_eq!(tri.dofs_per_node(), 2);
        assert_eq!(tri.n_dofs(), 6);
    }

    #[test]
    fn test_tri3_plane_strain_stiffness_symmetric() {
        let tri = Tri3PlaneStrain::new(1.0);
        let coords = unit_right_triangle();
        let mat = Material::steel();

        let k = tri.stiffness(&coords, &mat);

        assert_eq!(k.nrows(), 6);
        assert_eq!(k.ncols(), 6);
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(k[(i, j)], k[(j, i)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_tri3_plane_strain_stiffness_positive_diagonal() {
        let tri = Tri3PlaneStrain::new(1.0);
        let coords = unit_right_triangle();
        let mat = Material::steel();

        let k = tri.stiffness(&coords, &mat);

        for i in 0..6 {
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
    fn test_tri3_plane_strain_rigid_body_modes() {
        let tri = Tri3PlaneStrain::new(1.0);
        let coords = unit_right_triangle();
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = tri.stiffness(&coords, &mat);

        // Pure x-translation should produce zero force
        let u_x = nalgebra::DVector::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let f_x = &k * &u_x;
        assert_relative_eq!(f_x.norm(), 0.0, epsilon = 1e-12);

        // Pure y-translation
        let u_y = nalgebra::DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let f_y = &k * &u_y;
        assert_relative_eq!(f_y.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_tri3_plane_strain_constant_strain_patch() {
        let tri = Tri3PlaneStrain::new(1.0);
        let coords = unit_right_triangle();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Impose uniform ε_xx = 0.001: u = 0.001 * x
        // Node 0 (0,0): u=0
        // Node 1 (1,0): u=0.001
        // Node 2 (0,1): u=0
        let displacements = [0.0, 0.0, 0.001, 0.0, 0.0, 0.0];

        let stresses = tri.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 1);

        let stress = &stresses[0];
        let d = mat.constitutive_plane_strain();

        // Expected σ_xx = D[0,0] * ε_xx
        let expected_sigma_xx = d[(0, 0)] * 0.001;
        let expected_sigma_yy = d[(1, 0)] * 0.001;

        assert_relative_eq!(stress.0[0], expected_sigma_xx, epsilon = 1e-3);
        assert_relative_eq!(stress.0[1], expected_sigma_yy, epsilon = 1e-3);
        // For plane strain, σ_zz = ν(σ_x + σ_y) ≠ 0
        let expected_sigma_zz = mat.poissons_ratio * (stress.0[0] + stress.0[1]);
        assert_relative_eq!(stress.0[2], expected_sigma_zz, epsilon = 1e-3);
        assert_relative_eq!(stress.0[3], 0.0, epsilon = 1e-3); // τ_xy = 0
    }

    #[test]
    fn test_tri3_plane_strain_has_sigma_zz() {
        // Key difference from plane stress: σ_zz ≠ 0
        let tri = Tri3PlaneStrain::new(1.0);
        let coords = unit_right_triangle();
        let mat = Material::new(1e6, 0.3).unwrap();

        // Apply biaxial strain
        let displacements = [0.0, 0.0, 0.001, 0.001, 0.0, 0.001];

        let stresses = tri.stress(&coords, &displacements, &mat);
        let stress = &stresses[0];

        // σ_zz should be non-zero for plane strain
        // σ_zz = ν(σ_x + σ_y)
        let expected_sigma_zz = mat.poissons_ratio * (stress.0[0] + stress.0[1]);
        assert_relative_eq!(stress.0[2], expected_sigma_zz, epsilon = 1e-6);
        assert!(stress.0[2].abs() > 1e-6, "σ_zz should be non-zero for plane strain");
    }

    // === Quad4PlaneStrain Tests ===

    #[test]
    fn test_quad4_plane_strain_volume() {
        let quad = Quad4PlaneStrain::new(0.1);
        let coords = unit_square_quad();
        let vol = quad.volume(&coords);
        // Volume = Area * thickness = 1.0 * 0.1 = 0.1
        assert_relative_eq!(vol, 0.1, epsilon = 1e-14);
    }

    #[test]
    fn test_quad4_plane_strain_node_count() {
        let quad = Quad4PlaneStrain::new(1.0);
        assert_eq!(quad.n_nodes(), 4);
        assert_eq!(quad.dofs_per_node(), 2);
        assert_eq!(quad.n_dofs(), 8);
    }

    #[test]
    fn test_quad4_plane_strain_stiffness_symmetric() {
        let quad = Quad4PlaneStrain::new(1.0);
        let coords = unit_square_quad();
        let mat = Material::steel();

        let k = quad.stiffness(&coords, &mat);

        assert_eq!(k.nrows(), 8);
        assert_eq!(k.ncols(), 8);
        for i in 0..8 {
            for j in 0..8 {
                assert_relative_eq!(k[(i, j)], k[(j, i)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_quad4_plane_strain_stiffness_positive_diagonal() {
        let quad = Quad4PlaneStrain::new(1.0);
        let coords = unit_square_quad();
        let mat = Material::steel();

        let k = quad.stiffness(&coords, &mat);

        for i in 0..8 {
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
    fn test_quad4_plane_strain_rigid_body_modes() {
        let quad = Quad4PlaneStrain::new(1.0);
        let coords = unit_square_quad();
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = quad.stiffness(&coords, &mat);

        // Pure x-translation: u = [1,0,1,0,1,0,1,0]
        let u_x = nalgebra::DVector::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let f_x = &k * &u_x;
        assert_relative_eq!(f_x.norm(), 0.0, epsilon = 1e-12);

        // Pure y-translation: u = [0,1,0,1,0,1,0,1]
        let u_y = nalgebra::DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let f_y = &k * &u_y;
        assert_relative_eq!(f_y.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_quad4_plane_strain_constant_strain_patch() {
        let quad = Quad4PlaneStrain::new(1.0);
        let coords = unit_square_quad();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Impose uniform ε_xx = 0.001: u = 0.001 * x
        // Node 1 (0,0): u=0
        // Node 2 (1,0): u=0.001
        // Node 3 (1,1): u=0.001
        // Node 4 (0,1): u=0
        let displacements = [
            0.0, 0.0,    // Node 1
            0.001, 0.0,  // Node 2
            0.001, 0.0,  // Node 3
            0.0, 0.0,    // Node 4
        ];

        let stresses = quad.stress(&coords, &displacements, &mat);
        // 4 integration points
        assert_eq!(stresses.len(), 4);

        let d = mat.constitutive_plane_strain();
        let expected_sigma_xx = d[(0, 0)] * 0.001;
        let expected_sigma_yy = d[(1, 0)] * 0.001;

        // All integration points should have same stress (constant strain field)
        for stress in &stresses {
            assert_relative_eq!(stress.0[0], expected_sigma_xx, epsilon = 1e-3);
            assert_relative_eq!(stress.0[1], expected_sigma_yy, epsilon = 1e-3);
            // For plane strain, σ_zz = ν(σ_x + σ_y)
            let expected_sigma_zz = mat.poissons_ratio * (stress.0[0] + stress.0[1]);
            assert_relative_eq!(stress.0[2], expected_sigma_zz, epsilon = 1e-3);
            assert_relative_eq!(stress.0[3], 0.0, epsilon = 1e-3); // τ_xy = 0
        }
    }

    #[test]
    fn test_quad4_plane_strain_shear_strain() {
        let quad = Quad4PlaneStrain::new(1.0);
        let coords = unit_square_quad();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Impose pure shear: u = γ/2 * y, v = γ/2 * x where γ = 0.002
        // This gives γ_xy = ∂u/∂y + ∂v/∂x = 0.001 + 0.001 = 0.002
        let gamma = 0.002;
        let displacements = [
            0.0, 0.0,                      // Node 1: (0,0)
            0.0, gamma / 2.0,              // Node 2: (1,0) -> v = γ/2 * 1 = 0.001
            gamma / 2.0, gamma / 2.0,      // Node 3: (1,1) -> u = 0.001, v = 0.001
            gamma / 2.0, 0.0,              // Node 4: (0,1) -> u = γ/2 * 1 = 0.001
        ];

        let stresses = quad.stress(&coords, &displacements, &mat);

        let d = mat.constitutive_plane_strain();
        // For pure shear, σ_xx = σ_yy = 0, τ_xy = G * γ_xy = D[2,2] * γ
        let expected_tau_xy = d[(2, 2)] * gamma;

        for stress in &stresses {
            assert_relative_eq!(stress.0[0], 0.0, epsilon = 1e-3); // σ_xx = 0
            assert_relative_eq!(stress.0[1], 0.0, epsilon = 1e-3); // σ_yy = 0
            // σ_zz = ν(σ_x + σ_y) = 0 for pure shear
            assert_relative_eq!(stress.0[2], 0.0, epsilon = 1e-3);
            assert_relative_eq!(stress.0[3], expected_tau_xy, epsilon = 1e-3); // τ_xy
        }
    }

    #[test]
    fn test_quad4_plane_strain_has_sigma_zz() {
        // Key difference from plane stress: σ_zz ≠ 0 for normal strains
        let quad = Quad4PlaneStrain::new(1.0);
        let coords = unit_square_quad();
        let mat = Material::new(1e6, 0.3).unwrap();

        // Apply biaxial compression
        let displacements = [
            0.0, 0.0,
            -0.001, 0.0,
            -0.001, -0.001,
            0.0, -0.001,
        ];

        let stresses = quad.stress(&coords, &displacements, &mat);
        for stress in &stresses {
            // σ_zz should be non-zero for plane strain
            let expected_sigma_zz = mat.poissons_ratio * (stress.0[0] + stress.0[1]);
            assert_relative_eq!(stress.0[2], expected_sigma_zz, epsilon = 1e-6);
            assert!(stress.0[2].abs() > 1e-6, "σ_zz should be non-zero for plane strain");
        }
    }

    #[test]
    fn test_quad4_plane_strain_non_rectangular() {
        // Test with a non-rectangular quad (parallelogram)
        let quad = Quad4PlaneStrain::new(1.0);
        let coords = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.5, 1.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let mat = Material::steel();

        // Should still produce symmetric stiffness
        let k = quad.stiffness(&coords, &mat);
        for i in 0..8 {
            for j in 0..8 {
                // Use relative epsilon for large values
                let max_val = k[(i, j)].abs().max(k[(j, i)].abs());
                let rel_eps = if max_val > 1.0 { 1e-10 * max_val } else { 1e-10 };
                assert_relative_eq!(k[(i, j)], k[(j, i)], epsilon = rel_eps);
            }
        }

        // Volume = Area * t = 1.0 * 1.0 = 1.0 (parallelogram with base 1, height 1)
        let vol = quad.volume(&coords);
        assert_relative_eq!(vol, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quad4_plane_strain_shape_functions_sum_to_one() {
        // Shape functions should sum to 1 at any point
        let test_points = [
            (0.0, 0.0),
            (0.5, 0.5),
            (-0.5, 0.5),
            (0.0, 1.0),
            (-1.0, -1.0),
        ];
        for (xi, eta) in &test_points {
            let n = Quad4PlaneStrain::shape_functions(*xi, *eta);
            let sum: f64 = n.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_quad4_plane_strain_shape_functions_at_nodes() {
        // Shape function N_i should be 1 at node i and 0 at other nodes
        for (i, (xi_i, eta_i)) in Quad4PlaneStrain::NODE_COORDS.iter().enumerate() {
            let n = Quad4PlaneStrain::shape_functions(*xi_i, *eta_i);
            for j in 0..4 {
                if i == j {
                    assert_relative_eq!(n[j], 1.0, epsilon = 1e-14);
                } else {
                    assert_relative_eq!(n[j], 0.0, epsilon = 1e-14);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "Thickness must be positive")]
    fn test_tri3_plane_strain_invalid_thickness() {
        Tri3PlaneStrain::new(0.0);
    }

    #[test]
    #[should_panic(expected = "Thickness must be positive")]
    fn test_quad4_plane_strain_invalid_thickness() {
        Quad4PlaneStrain::new(-1.0);
    }

    // === Comparison Tests: Plane Strain vs Plane Stress ===

    #[test]
    fn test_plane_strain_stiffer_than_plane_stress() {
        use crate::element::plane_stress::{Tri3 as Tri3Stress, Quad4 as Quad4Stress};

        let coords_tri = unit_right_triangle();
        let coords_quad = unit_square_quad();
        let mat = Material::new(1e6, 0.3).unwrap();

        // Plane strain should be stiffer than plane stress (constrained in z)
        let tri_strain = Tri3PlaneStrain::new(1.0);
        let tri_stress = Tri3Stress::new(1.0);

        let k_strain = tri_strain.stiffness(&coords_tri, &mat);
        let k_stress = tri_stress.stiffness(&coords_tri, &mat);

        // Diagonal terms of plane strain should be >= plane stress
        for i in 0..6 {
            assert!(
                k_strain[(i, i)] >= k_stress[(i, i)],
                "K_strain[{0},{0}] = {1} should be >= K_stress[{0},{0}] = {2}",
                i,
                k_strain[(i, i)],
                k_stress[(i, i)]
            );
        }

        // Same for Quad4
        let quad_strain = Quad4PlaneStrain::new(1.0);
        let quad_stress = Quad4Stress::new(1.0);

        let k_strain = quad_strain.stiffness(&coords_quad, &mat);
        let k_stress = quad_stress.stiffness(&coords_quad, &mat);

        for i in 0..8 {
            assert!(
                k_strain[(i, i)] >= k_stress[(i, i)],
                "K_strain[{0},{0}] = {1} should be >= K_stress[{0},{0}] = {2}",
                i,
                k_strain[(i, i)],
                k_stress[(i, i)]
            );
        }
    }
}
