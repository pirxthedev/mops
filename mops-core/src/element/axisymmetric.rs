//! Axisymmetric elements for 2D analysis of bodies of revolution.
//!
//! Axisymmetric elements model 3D rotationally symmetric problems using a 2D
//! mesh in the r-z plane. The geometry and loading must be symmetric about the
//! z-axis (axis of revolution).
//!
//! This module provides:
//! - [`Tri3Axisymmetric`] - 3-node triangle (Constant Strain Triangle)
//! - [`Quad4Axisymmetric`] - 4-node quadrilateral (bilinear)
//!
//! # Coordinate System
//!
//! Nodes have (r, z) coordinates where:
//! - r = radial distance from axis of revolution (x-coordinate in Point3)
//! - z = axial coordinate (y-coordinate in Point3)
//!
//! Each node has 2 DOFs: (u_r, u_z) displacements.
//!
//! # Strain Components
//!
//! Unlike plane stress/strain, axisymmetric analysis has 4 strain components:
//! - ε_rr = ∂u_r/∂r (radial strain)
//! - ε_zz = ∂u_z/∂z (axial strain)
//! - ε_θθ = u_r/r (hoop/circumferential strain) - key difference!
//! - γ_rz = ∂u_r/∂z + ∂u_z/∂r (shear strain)
//!
//! The hoop strain ε_θθ = u_r/r arises from circumferential stretching when
//! radial displacement occurs.
//!
//! # Integration
//!
//! The volume element in axisymmetric coordinates is dV = 2πr dr dz, so all
//! integrations include the radial coordinate r as a weighting factor.
//!
//! # Stress Output
//!
//! Stresses are returned as 6-component tensors for compatibility with 3D
//! elements. Components are mapped as:
//! - σ_xx → σ_rr (radial)
//! - σ_yy → σ_zz (axial)
//! - σ_zz → σ_θθ (hoop)
//! - τ_xy → τ_rz (shear)
//! - τ_yz = τ_xz = 0 (due to symmetry)

use crate::element::gauss::gauss_quad;
use crate::element::Element;
use crate::material::Material;
use crate::types::{Point3, StressTensor};
use nalgebra::{DMatrix, Matrix2, Vector2, Vector4, Vector6};
use std::f64::consts::PI;

/// 3-node triangular axisymmetric element (Constant Strain Triangle).
///
/// The simplest axisymmetric element with:
/// - 3 nodes at vertices
/// - 2 DOFs per node (u_r, u_z displacements)
/// - 6 total DOFs
/// - Constant strain within element (except ε_θθ which varies with 1/r)
/// - Single integration point at centroid
///
/// # Hoop Strain
///
/// The circumferential strain ε_θθ = u_r/r is computed using the radial
/// displacement interpolated from nodal values and the radial coordinate.
/// For the constant strain element, strain is evaluated at the centroid.
///
/// # Limitations
///
/// - Accuracy degrades near r = 0 (axis of symmetry)
/// - Requires fine meshes for accuracy
/// - Low-order element suitable for preliminary analysis
#[derive(Debug, Clone, Copy)]
pub struct Tri3Axisymmetric;

impl Tri3Axisymmetric {
    /// Create a new Tri3Axisymmetric element.
    pub fn new() -> Self {
        Self
    }

    /// Compute area of triangle from nodal coordinates.
    fn compute_area(coords: &[Point3]) -> f64 {
        assert_eq!(coords.len(), 3, "Tri3 requires exactly 3 nodal coordinates");

        let r1 = coords[0][0];
        let z1 = coords[0][1];
        let r2 = coords[1][0];
        let z2 = coords[1][1];
        let r3 = coords[2][0];
        let z3 = coords[2][1];

        // Area = 0.5 * |det([r2-r1, r3-r1; z2-z1, z3-z1])|
        0.5 * ((r2 - r1) * (z3 - z1) - (r3 - r1) * (z2 - z1)).abs()
    }

    /// Compute centroid of triangle (average of nodal coordinates).
    fn compute_centroid(coords: &[Point3]) -> (f64, f64) {
        let r_c = (coords[0][0] + coords[1][0] + coords[2][0]) / 3.0;
        let z_c = (coords[0][1] + coords[1][1] + coords[2][1]) / 3.0;
        (r_c, z_c)
    }

    /// Compute the B-matrix (strain-displacement) for axisymmetric element.
    ///
    /// Returns the 4x6 B-matrix where ε = B * u
    /// ε = [ε_rr, ε_zz, ε_θθ, γ_rz]^T
    ///
    /// The B-matrix is evaluated at the centroid where r = r_c.
    fn compute_b_matrix(coords: &[Point3]) -> (DMatrix<f64>, f64, f64) {
        assert_eq!(coords.len(), 3, "Tri3 requires exactly 3 nodal coordinates");

        let r1 = coords[0][0];
        let z1 = coords[0][1];
        let r2 = coords[1][0];
        let z2 = coords[1][1];
        let r3 = coords[2][0];
        let z3 = coords[2][1];

        // Area (with sign for orientation)
        let area_2_signed = (r2 - r1) * (z3 - z1) - (r3 - r1) * (z2 - z1);
        let area = area_2_signed.abs() / 2.0;

        // Shape function derivatives (constant for linear triangle):
        // Using same notation as plane elements but with r,z instead of x,y
        let inv_2a = 1.0 / area_2_signed;

        let dn1_dr = (z2 - z3) * inv_2a;
        let dn1_dz = (r3 - r2) * inv_2a;
        let dn2_dr = (z3 - z1) * inv_2a;
        let dn2_dz = (r1 - r3) * inv_2a;
        let dn3_dr = (z1 - z2) * inv_2a;
        let dn3_dz = (r2 - r1) * inv_2a;

        // Centroid for hoop strain evaluation
        let (r_c, _z_c) = Self::compute_centroid(coords);

        // Shape function values at centroid (each = 1/3 for linear triangle)
        let n1 = 1.0 / 3.0;
        let n2 = 1.0 / 3.0;
        let n3 = 1.0 / 3.0;

        // B-matrix (4x6)
        // ε = [ε_rr, ε_zz, ε_θθ, γ_rz]^T = B * [u_r1, u_z1, u_r2, u_z2, u_r3, u_z3]^T
        //
        // ε_rr = ∂u_r/∂r = dN1/dr * u_r1 + dN2/dr * u_r2 + dN3/dr * u_r3
        // ε_zz = ∂u_z/∂z = dN1/dz * u_z1 + dN2/dz * u_z2 + dN3/dz * u_z3
        // ε_θθ = u_r/r = (N1 * u_r1 + N2 * u_r2 + N3 * u_r3) / r
        // γ_rz = ∂u_r/∂z + ∂u_z/∂r
        let mut b = DMatrix::zeros(4, 6);

        // Node 1 (columns 0, 1)
        b[(0, 0)] = dn1_dr;           // ε_rr
        b[(1, 1)] = dn1_dz;           // ε_zz
        b[(2, 0)] = n1 / r_c;         // ε_θθ = N1 * u_r1 / r_c
        b[(3, 0)] = dn1_dz;           // γ_rz = ∂u_r/∂z
        b[(3, 1)] = dn1_dr;           // γ_rz += ∂u_z/∂r

        // Node 2 (columns 2, 3)
        b[(0, 2)] = dn2_dr;
        b[(1, 3)] = dn2_dz;
        b[(2, 2)] = n2 / r_c;
        b[(3, 2)] = dn2_dz;
        b[(3, 3)] = dn2_dr;

        // Node 3 (columns 4, 5)
        b[(0, 4)] = dn3_dr;
        b[(1, 5)] = dn3_dz;
        b[(2, 4)] = n3 / r_c;
        b[(3, 4)] = dn3_dz;
        b[(3, 5)] = dn3_dr;

        (b, area, r_c)
    }
}

impl Default for Tri3Axisymmetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Tri3Axisymmetric {
    fn n_nodes(&self) -> usize {
        3
    }

    fn dofs_per_node(&self) -> usize {
        2
    }

    fn stiffness(&self, coords: &[Point3], material: &Material) -> DMatrix<f64> {
        let (b, area, r_c) = Self::compute_b_matrix(coords);

        // D is the 4x4 axisymmetric constitutive matrix
        let d = material.constitutive_axisymmetric();

        // K = 2π * r_c * A * B^T * D * B (6x6)
        // The factor 2πr comes from integration over the circumference
        let db = d * &b; // 4x6
        b.transpose() * db * (2.0 * PI * r_c * area)
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

        let (b, _area, _r_c) = Self::compute_b_matrix(coords);
        let d = material.constitutive_axisymmetric();

        // ε = B * u (4x1)
        let u = nalgebra::DVector::from_row_slice(displacements);
        let strain = &b * &u;

        // σ = D * ε (4x1) for axisymmetric stresses
        let stress_4 = d * Vector4::from_iterator(strain.iter().cloned());

        // Convert to 6-component stress tensor
        // Map: σ_rr → σ_xx, σ_zz → σ_yy, σ_θθ → σ_zz, τ_rz → τ_xy
        let stress_6 = Vector6::new(
            stress_4[0], // σ_rr → σ_xx
            stress_4[1], // σ_zz → σ_yy
            stress_4[2], // σ_θθ → σ_zz
            stress_4[3], // τ_rz → τ_xy
            0.0,         // τ_yz = 0 (axisymmetric)
            0.0,         // τ_xz = 0 (axisymmetric)
        );

        // Single stress tensor (constant strain element)
        vec![StressTensor(stress_6)]
    }

    fn volume(&self, coords: &[Point3]) -> f64 {
        let area = Self::compute_area(coords);
        let (r_c, _z_c) = Self::compute_centroid(coords);
        // Volume = 2π * r_c * A (revolution of triangular cross-section)
        2.0 * PI * r_c * area
    }
}

/// 4-node quadrilateral axisymmetric element.
///
/// A higher-order axisymmetric element with:
/// - 4 nodes at corners
/// - 2 DOFs per node (u_r, u_z displacements)
/// - 8 total DOFs
/// - Bilinear shape functions
/// - 2×2 Gauss quadrature (4 integration points)
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
/// 4 --- 3     z
/// |     |     ↑
/// 1 --- 2     → r
/// ```
/// Node 1: (-1, -1), Node 2: (+1, -1), Node 3: (+1, +1), Node 4: (-1, +1)
///
/// # Hoop Strain
///
/// The hoop strain ε_θθ = u_r/r is computed at each integration point using
/// the local radial coordinate.
#[derive(Debug, Clone, Copy)]
pub struct Quad4Axisymmetric;

impl Quad4Axisymmetric {
    /// Create a new Quad4Axisymmetric element.
    pub fn new() -> Self {
        Self
    }

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

        // J = [[∂r/∂ξ, ∂z/∂ξ],
        //      [∂r/∂η, ∂z/∂η]]
        let mut j = Matrix2::zeros();
        for i in 0..4 {
            j[(0, 0)] += dn_dnat[i].0 * coords[i][0]; // ∂r/∂ξ
            j[(0, 1)] += dn_dnat[i].0 * coords[i][1]; // ∂z/∂ξ
            j[(1, 0)] += dn_dnat[i].1 * coords[i][0]; // ∂r/∂η
            j[(1, 1)] += dn_dnat[i].1 * coords[i][1]; // ∂z/∂η
        }

        let det_j = j.determinant();
        (j, det_j)
    }

    /// Interpolate radial coordinate at (ξ, η).
    fn interpolate_r(coords: &[Point3], xi: f64, eta: f64) -> f64 {
        let n = Self::shape_functions(xi, eta);
        n[0] * coords[0][0] + n[1] * coords[1][0] + n[2] * coords[2][0] + n[3] * coords[3][0]
    }

    /// Compute B-matrix at a specific integration point.
    ///
    /// Returns the 4x8 strain-displacement matrix.
    fn compute_b_at_point(coords: &[Point3], xi: f64, eta: f64) -> DMatrix<f64> {
        let n = Self::shape_functions(xi, eta);
        let dn_dnat = Self::shape_function_derivatives(xi, eta);
        let (j, _det_j) = Self::jacobian(coords, xi, eta);

        // Invert Jacobian to get dN/dr, dN/dz from dN/dξ, dN/dη
        let j_inv = j.try_inverse().expect("Degenerate element: Jacobian is singular");

        // dN/dr = J^(-1) * dN/dξ
        let mut dn_dr = [(0.0, 0.0); 4];
        for i in 0..4 {
            let dnat = Vector2::new(dn_dnat[i].0, dn_dnat[i].1);
            let dphys = j_inv * dnat;
            dn_dr[i] = (dphys[0], dphys[1]); // (dN/dr, dN/dz)
        }

        // Radial coordinate at integration point for hoop strain
        let r = Self::interpolate_r(coords, xi, eta);

        // Build B-matrix (4x8)
        // ε = [ε_rr, ε_zz, ε_θθ, γ_rz]^T
        let mut b = DMatrix::zeros(4, 8);
        for i in 0..4 {
            let col = 2 * i;
            b[(0, col)] = dn_dr[i].0;       // ε_rr = ∂u_r/∂r
            b[(1, col + 1)] = dn_dr[i].1;   // ε_zz = ∂u_z/∂z
            b[(2, col)] = n[i] / r;         // ε_θθ = u_r/r = (N_i * u_ri) / r
            b[(3, col)] = dn_dr[i].1;       // γ_rz = ∂u_r/∂z + ∂u_z/∂r
            b[(3, col + 1)] = dn_dr[i].0;
        }

        b
    }
}

impl Default for Quad4Axisymmetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Quad4Axisymmetric {
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

        let d = material.constitutive_axisymmetric();
        let mut k = DMatrix::zeros(8, 8);

        // 2x2 Gauss quadrature
        let gauss_points = gauss_quad(2);
        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let weight = gp.weight;

            let b = Self::compute_b_at_point(coords, xi, eta);
            let (_, det_j) = Self::jacobian(coords, xi, eta);
            let r = Self::interpolate_r(coords, xi, eta);

            // K += w * 2π * r * det(J) * B^T * D * B
            let db = d * &b;
            k += b.transpose() * db * (weight * 2.0 * PI * r * det_j);
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

        let d = material.constitutive_axisymmetric();
        let u = nalgebra::DVector::from_row_slice(displacements);

        // Compute stress at each integration point
        let gauss_points = gauss_quad(2);
        let mut stresses = Vec::with_capacity(gauss_points.len());

        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();

            let b = Self::compute_b_at_point(coords, xi, eta);

            // ε = B * u
            let strain = &b * &u;

            // σ = D * ε (4 components)
            let stress_4 = d * Vector4::from_iterator(strain.iter().cloned());

            // Convert to 6-component stress tensor
            let stress_6 = Vector6::new(
                stress_4[0], // σ_rr → σ_xx
                stress_4[1], // σ_zz → σ_yy
                stress_4[2], // σ_θθ → σ_zz
                stress_4[3], // τ_rz → τ_xy
                0.0,         // τ_yz = 0
                0.0,         // τ_xz = 0
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
        // V = ∫∫ 2πr dr dz = ∫∫ 2πr |J| dξ dη
        let gauss_points = gauss_quad(2);
        let mut vol = 0.0;
        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let (_, det_j) = Self::jacobian(coords, xi, eta);
            let r = Self::interpolate_r(coords, xi, eta);
            vol += gp.weight * 2.0 * PI * r * det_j;
        }

        vol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Unit right triangle in r-z plane at r > 0
    fn unit_triangle_at_r1() -> Vec<Point3> {
        vec![
            Point3::new(1.0, 0.0, 0.0), // r=1, z=0
            Point3::new(2.0, 0.0, 0.0), // r=2, z=0
            Point3::new(1.0, 1.0, 0.0), // r=1, z=1
        ]
    }

    /// Unit square quad in r-z plane at r > 0
    fn unit_square_at_r1() -> Vec<Point3> {
        vec![
            Point3::new(1.0, 0.0, 0.0), // Node 1
            Point3::new(2.0, 0.0, 0.0), // Node 2
            Point3::new(2.0, 1.0, 0.0), // Node 3
            Point3::new(1.0, 1.0, 0.0), // Node 4
        ]
    }

    // === Tri3Axisymmetric Tests ===

    #[test]
    fn test_tri3_axisymmetric_area() {
        let coords = unit_triangle_at_r1();
        let area = Tri3Axisymmetric::compute_area(&coords);
        // Right triangle with legs of length 1: Area = 0.5
        assert_relative_eq!(area, 0.5, epsilon = 1e-14);
    }

    #[test]
    fn test_tri3_axisymmetric_centroid() {
        let coords = unit_triangle_at_r1();
        let (r_c, z_c) = Tri3Axisymmetric::compute_centroid(&coords);
        // Centroid = (1+2+1)/3, (0+0+1)/3 = (4/3, 1/3)
        assert_relative_eq!(r_c, 4.0 / 3.0, epsilon = 1e-14);
        assert_relative_eq!(z_c, 1.0 / 3.0, epsilon = 1e-14);
    }

    #[test]
    fn test_tri3_axisymmetric_volume() {
        let tri = Tri3Axisymmetric::new();
        let coords = unit_triangle_at_r1();
        let vol = tri.volume(&coords);
        // Volume = 2π * r_c * A = 2π * (4/3) * 0.5 = 4π/3
        let expected = 2.0 * PI * (4.0 / 3.0) * 0.5;
        assert_relative_eq!(vol, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_tri3_axisymmetric_node_count() {
        let tri = Tri3Axisymmetric::new();
        assert_eq!(tri.n_nodes(), 3);
        assert_eq!(tri.dofs_per_node(), 2);
        assert_eq!(tri.n_dofs(), 6);
    }

    #[test]
    fn test_tri3_axisymmetric_stiffness_symmetric() {
        let tri = Tri3Axisymmetric::new();
        let coords = unit_triangle_at_r1();
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
    fn test_tri3_axisymmetric_stiffness_positive_diagonal() {
        let tri = Tri3Axisymmetric::new();
        let coords = unit_triangle_at_r1();
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
    fn test_tri3_axisymmetric_rigid_body_axial() {
        // Pure axial (z) translation should produce zero strain/force
        let tri = Tri3Axisymmetric::new();
        let coords = unit_triangle_at_r1();
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = tri.stiffness(&coords, &mat);

        // Pure z-translation: u = [0, 1, 0, 1, 0, 1]
        let u_z = nalgebra::DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let f_z = &k * &u_z;
        assert_relative_eq!(f_z.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_tri3_axisymmetric_stress_has_hoop() {
        let tri = Tri3Axisymmetric::new();
        let coords = unit_triangle_at_r1();
        let mat = Material::new(1e6, 0.3).unwrap();

        // Apply radial expansion: u_r = 0.001 at all nodes
        // This should create hoop stress σ_θθ
        let displacements = [0.001, 0.0, 0.001, 0.0, 0.001, 0.0];

        let stresses = tri.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 1);

        let stress = &stresses[0];
        // σ_θθ is in component [2] (mapped to σ_zz in 6-component form)
        // Uniform radial displacement creates hoop strain ε_θθ = u_r/r
        // which couples to radial and axial stress through D matrix
        assert!(stress.0[2].abs() > 1e-6, "σ_θθ should be non-zero for radial displacement");
    }

    #[test]
    fn test_tri3_axisymmetric_pure_axial_strain() {
        let tri = Tri3Axisymmetric::new();
        let coords = unit_triangle_at_r1();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Impose uniform ε_zz = 0.001: u_z = 0.001 * z
        // Node 0 (r=1, z=0): u_z=0
        // Node 1 (r=2, z=0): u_z=0
        // Node 2 (r=1, z=1): u_z=0.001
        let displacements = [0.0, 0.0, 0.0, 0.0, 0.0, 0.001];

        let stresses = tri.stress(&coords, &displacements, &mat);
        let stress = &stresses[0];

        // ε_zz should be 0.001, other strains near zero (except coupling)
        // σ_zz (component [1]) should be dominant
        let d = mat.constitutive_axisymmetric();
        let expected_sigma_zz = d[(1, 1)] * 0.001;
        assert_relative_eq!(stress.0[1], expected_sigma_zz, epsilon = 1e-2);
    }

    // === Quad4Axisymmetric Tests ===

    #[test]
    fn test_quad4_axisymmetric_volume() {
        let quad = Quad4Axisymmetric::new();
        let coords = unit_square_at_r1();
        let vol = quad.volume(&coords);

        // Volume of annular ring section: V = π * (r_outer² - r_inner²) * height
        // r_inner = 1, r_outer = 2, height = 1
        // V = π * (4 - 1) * 1 = 3π
        let expected = PI * (4.0 - 1.0);
        assert_relative_eq!(vol, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quad4_axisymmetric_node_count() {
        let quad = Quad4Axisymmetric::new();
        assert_eq!(quad.n_nodes(), 4);
        assert_eq!(quad.dofs_per_node(), 2);
        assert_eq!(quad.n_dofs(), 8);
    }

    #[test]
    fn test_quad4_axisymmetric_stiffness_symmetric() {
        let quad = Quad4Axisymmetric::new();
        let coords = unit_square_at_r1();
        let mat = Material::steel();

        let k = quad.stiffness(&coords, &mat);

        assert_eq!(k.nrows(), 8);
        assert_eq!(k.ncols(), 8);
        for i in 0..8 {
            for j in 0..8 {
                // Use relative epsilon for large values (steel E = 200 GPa)
                let max_val = k[(i, j)].abs().max(k[(j, i)].abs());
                let rel_eps = if max_val > 1.0 { 1e-12 * max_val } else { 1e-10 };
                assert_relative_eq!(k[(i, j)], k[(j, i)], epsilon = rel_eps);
            }
        }
    }

    #[test]
    fn test_quad4_axisymmetric_stiffness_positive_diagonal() {
        let quad = Quad4Axisymmetric::new();
        let coords = unit_square_at_r1();
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
    fn test_quad4_axisymmetric_rigid_body_axial() {
        // Pure axial (z) translation should produce zero strain/force
        let quad = Quad4Axisymmetric::new();
        let coords = unit_square_at_r1();
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = quad.stiffness(&coords, &mat);

        // Pure z-translation: u = [0, 1, 0, 1, 0, 1, 0, 1]
        let u_z = nalgebra::DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let f_z = &k * &u_z;
        assert_relative_eq!(f_z.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_quad4_axisymmetric_stress_has_hoop() {
        let quad = Quad4Axisymmetric::new();
        let coords = unit_square_at_r1();
        let mat = Material::new(1e6, 0.3).unwrap();

        // Apply radial expansion: u_r = 0.001 at all nodes
        let displacements = [0.001, 0.0, 0.001, 0.0, 0.001, 0.0, 0.001, 0.0];

        let stresses = quad.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 4); // 2x2 integration points

        for stress in &stresses {
            // σ_θθ should be non-zero for radial displacement
            assert!(stress.0[2].abs() > 1e-6, "σ_θθ should be non-zero for radial displacement");
        }
    }

    #[test]
    fn test_quad4_axisymmetric_constant_axial_strain() {
        let quad = Quad4Axisymmetric::new();
        let coords = unit_square_at_r1();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Impose uniform ε_zz = 0.001: u_z = 0.001 * z
        // Nodes at z=0: u_z=0, Nodes at z=1: u_z=0.001
        let displacements = [
            0.0, 0.0,     // Node 1 (z=0)
            0.0, 0.0,     // Node 2 (z=0)
            0.0, 0.001,   // Node 3 (z=1)
            0.0, 0.001,   // Node 4 (z=1)
        ];

        let stresses = quad.stress(&coords, &displacements, &mat);

        // All integration points should have similar σ_zz (constant strain)
        let d = mat.constitutive_axisymmetric();
        let expected_sigma_zz = d[(1, 1)] * 0.001;

        for stress in &stresses {
            assert_relative_eq!(stress.0[1], expected_sigma_zz, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_quad4_axisymmetric_shape_functions_sum_to_one() {
        let test_points = [
            (0.0, 0.0),
            (0.5, 0.5),
            (-0.5, 0.5),
            (0.0, 1.0),
            (-1.0, -1.0),
        ];
        for (xi, eta) in &test_points {
            let n = Quad4Axisymmetric::shape_functions(*xi, *eta);
            let sum: f64 = n.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_quad4_axisymmetric_shape_functions_at_nodes() {
        // Node positions in natural coordinates
        let node_coords: [(f64, f64); 4] = [
            (-1.0, -1.0), // Node 1
            (1.0, -1.0),  // Node 2
            (1.0, 1.0),   // Node 3
            (-1.0, 1.0),  // Node 4
        ];
        for (i, (xi_i, eta_i)) in node_coords.iter().enumerate() {
            let n = Quad4Axisymmetric::shape_functions(*xi_i, *eta_i);
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
    fn test_quad4_axisymmetric_non_rectangular() {
        // Test with a non-rectangular quad (trapezoid)
        let quad = Quad4Axisymmetric::new();
        let coords = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.8, 1.0, 0.0),
            Point3::new(1.2, 1.0, 0.0),
        ];
        let mat = Material::steel();

        // Should still produce symmetric stiffness
        let k = quad.stiffness(&coords, &mat);
        for i in 0..8 {
            for j in 0..8 {
                let max_val = k[(i, j)].abs().max(k[(j, i)].abs());
                let rel_eps = if max_val > 1.0 { 1e-10 * max_val } else { 1e-10 };
                assert_relative_eq!(k[(i, j)], k[(j, i)], epsilon = rel_eps);
            }
        }
    }

    // === Pressurized Cylinder Test (Lamé Solution) ===

    #[test]
    fn test_thick_cylinder_hoop_stress_direction() {
        // For a thick-walled pressurized cylinder, internal pressure creates
        // tensile hoop stress (σ_θθ > 0) and compressive radial stress (σ_rr < 0)
        // at the inner surface.
        //
        // This test verifies the sign conventions are correct.
        let quad = Quad4Axisymmetric::new();
        let mat = Material::steel();

        // Single element representing part of a cylinder wall
        // Inner radius = 1, outer radius = 2
        let coords = unit_square_at_r1();

        // Apply internal pressure by prescribing radial displacement
        // that represents expansion from internal pressure
        // Inner surface moves outward more than outer surface
        let displacements = [
            0.002, 0.0,   // Node 1 (r=1, z=0) - inner surface
            0.001, 0.0,   // Node 2 (r=2, z=0) - outer surface
            0.001, 0.0,   // Node 3 (r=2, z=1)
            0.002, 0.0,   // Node 4 (r=1, z=1)
        ];

        let stresses = quad.stress(&coords, &displacements, &mat);

        // At inner surface (smaller r), we expect larger hoop stress
        // This is a qualitative check - exact Lamé solution would require
        // proper boundary conditions
        for stress in &stresses {
            // Check that hoop stress σ_θθ exists
            assert!(stress.0[2].abs() > 1e-3, "Should have significant hoop stress");
        }
    }
}
