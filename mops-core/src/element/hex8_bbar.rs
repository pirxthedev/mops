//! 8-node hexahedron with B-bar (Mean Dilatation) formulation.
//!
//! This element variant addresses both volumetric locking (nearly incompressible
//! materials, ν → 0.5) and shear locking by using element-averaged volumetric strain.
//!
//! # Background
//!
//! Volumetric locking occurs in nearly incompressible materials where the bulk modulus
//! becomes very large compared to the shear modulus. Standard elements can exhibit
//! artificial stiffness or even lock completely for ν → 0.5.
//!
//! The B-bar method mitigates this by:
//! 1. Computing the element-averaged volumetric B-matrix: B̄_vol = (1/V) ∫ B_vol dV
//! 2. Replacing local volumetric strain with element-averaged volumetric strain
//! 3. Computing the modified B-matrix: B̄ = B_dev + B̄_vol
//!
//! # Mathematical Formulation
//!
//! The standard B-matrix is split into volumetric and deviatoric parts:
//! ```text
//! B = B_vol + B_dev
//!
//! B_vol = I_vol * B  where I_vol = (1/3) * m * m^T, m = [1,1,1,0,0,0]^T
//! B_dev = I_dev * B  where I_dev = I - I_vol
//! ```
//!
//! The B-bar matrix uses the element-averaged volumetric part:
//! ```text
//! B̄_vol = (1/V) ∫ B_vol dV  (computed via 8-point integration)
//!
//! B̄ = B_dev + B̄_vol
//! ```
//!
//! The element stiffness is then:
//! ```text
//! K = ∫ B̄^T * D * B̄ dV  (8-point integration using B̄ at each point)
//! ```
//!
//! # Advantages Over Standard Hex8
//!
//! - Eliminates volumetric locking for nearly incompressible materials (ν → 0.5)
//! - Reduces shear locking in bending problems
//! - Passes all standard patch tests
//! - Mathematically rigorous (variational basis)
//!
//! # Comparison with Hex8SRI
//!
//! - Hex8SRI: Uses 1-point volumetric integration (simpler, good for bending)
//! - Hex8Bbar: Uses averaged volumetric strain (better for incompressible materials)
//!
//! # References
//!
//! 1. Hughes, T.J.R. "The Finite Element Method" (2000), Ch. 4
//! 2. Simo, J.C. and Rifai, M.S. "A class of mixed assumed strain methods and
//!    the method of incompatible modes" (1990)
//! 3. Belytschko et al. "Nonlinear Finite Elements for Continua and Structures"

use crate::element::gauss::gauss_hex;
use crate::element::Element;
use crate::material::Material;
use crate::types::{Point3, StressTensor};
use nalgebra::{DMatrix, Matrix3, Matrix6, Vector3, Vector6};

/// Natural coordinates for each of the 8 nodes.
/// Node i has natural coordinates (XI[i], ETA[i], ZETA[i]).
const XI: [f64; 8] = [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0];
const ETA: [f64; 8] = [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0];
const ZETA: [f64; 8] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];

/// 8-node hexahedral element with B-bar (mean dilatation) formulation.
///
/// Uses element-averaged volumetric strain to mitigate both volumetric locking
/// (nearly incompressible materials) and shear locking (bending).
#[derive(Debug, Clone, Copy, Default)]
pub struct Hex8Bbar;

impl Hex8Bbar {
    /// Create a new Hex8Bbar element.
    pub fn new() -> Self {
        Self
    }

    /// Compute shape functions at a point in natural coordinates.
    ///
    /// Returns N = [N_0, N_1, ..., N_7]
    #[cfg(test)]
    fn shape_functions(xi: f64, eta: f64, zeta: f64) -> [f64; 8] {
        let mut n = [0.0; 8];
        for i in 0..8 {
            n[i] = 0.125 * (1.0 + XI[i] * xi) * (1.0 + ETA[i] * eta) * (1.0 + ZETA[i] * zeta);
        }
        n
    }

    /// Compute shape function derivatives with respect to natural coordinates.
    ///
    /// Returns (dN/dξ, dN/dη, dN/dζ) for each node.
    fn shape_derivatives(xi: f64, eta: f64, zeta: f64) -> ([f64; 8], [f64; 8], [f64; 8]) {
        let mut dn_dxi = [0.0; 8];
        let mut dn_deta = [0.0; 8];
        let mut dn_dzeta = [0.0; 8];

        for i in 0..8 {
            let xi_i = XI[i];
            let eta_i = ETA[i];
            let zeta_i = ZETA[i];

            // dN_i/dξ = (1/8) * ξ_i * (1 + η_i*η) * (1 + ζ_i*ζ)
            dn_dxi[i] = 0.125 * xi_i * (1.0 + eta_i * eta) * (1.0 + zeta_i * zeta);

            // dN_i/dη = (1/8) * (1 + ξ_i*ξ) * η_i * (1 + ζ_i*ζ)
            dn_deta[i] = 0.125 * (1.0 + xi_i * xi) * eta_i * (1.0 + zeta_i * zeta);

            // dN_i/dζ = (1/8) * (1 + ξ_i*ξ) * (1 + η_i*η) * ζ_i
            dn_dzeta[i] = 0.125 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta) * zeta_i;
        }

        (dn_dxi, dn_deta, dn_dzeta)
    }

    /// Compute the Jacobian matrix at a point in natural coordinates.
    ///
    /// J = [∂x/∂ξ  ∂y/∂ξ  ∂z/∂ξ]
    ///     [∂x/∂η  ∂y/∂η  ∂z/∂η]
    ///     [∂x/∂ζ  ∂y/∂ζ  ∂z/∂ζ]
    fn jacobian(
        coords: &[Point3],
        dn_dxi: &[f64; 8],
        dn_deta: &[f64; 8],
        dn_dzeta: &[f64; 8],
    ) -> Matrix3<f64> {
        let mut j = Matrix3::zeros();

        for i in 0..8 {
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

    /// Compute the B-matrix (strain-displacement) at a point in natural coordinates.
    ///
    /// Returns the 6×24 B-matrix and the Jacobian determinant.
    fn compute_b_matrix(coords: &[Point3], xi: f64, eta: f64, zeta: f64) -> (DMatrix<f64>, f64) {
        let (dn_dxi, dn_deta, dn_dzeta) = Self::shape_derivatives(xi, eta, zeta);
        let j = Self::jacobian(coords, &dn_dxi, &dn_deta, &dn_dzeta);

        let det_j = j.determinant();
        assert!(
            det_j > 0.0,
            "Negative Jacobian determinant ({}) indicates inverted element",
            det_j
        );

        let j_inv = j.try_inverse().expect("Jacobian is singular");

        // Compute dN/dx = J^(-1) * dN/dξ for each node
        let mut dn_dx = [0.0; 8];
        let mut dn_dy = [0.0; 8];
        let mut dn_dz = [0.0; 8];

        for i in 0..8 {
            let dn_dnat = Vector3::new(dn_dxi[i], dn_deta[i], dn_dzeta[i]);
            let dn_dphys = j_inv * dn_dnat;
            dn_dx[i] = dn_dphys[0];
            dn_dy[i] = dn_dphys[1];
            dn_dz[i] = dn_dphys[2];
        }

        // Construct B-matrix (6×24)
        // ε = [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz]^T = B * u
        let mut b = DMatrix::zeros(6, 24);

        for i in 0..8 {
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

    /// Compute the volumetric projection matrix I_vol.
    ///
    /// I_vol = (1/3) * m * m^T where m = [1, 1, 1, 0, 0, 0]^T
    ///
    /// This 6×6 matrix projects a strain vector onto its volumetric component.
    fn volumetric_projector() -> Matrix6<f64> {
        // m = [1, 1, 1, 0, 0, 0]^T
        // I_vol = (1/3) * m * m^T
        // Result:
        // [1/3  1/3  1/3  0  0  0]
        // [1/3  1/3  1/3  0  0  0]
        // [1/3  1/3  1/3  0  0  0]
        // [0    0    0    0  0  0]
        // [0    0    0    0  0  0]
        // [0    0    0    0  0  0]
        let third = 1.0 / 3.0;
        #[rustfmt::skip]
        let i_vol = Matrix6::new(
            third, third, third, 0.0, 0.0, 0.0,
            third, third, third, 0.0, 0.0, 0.0,
            third, third, third, 0.0, 0.0, 0.0,
            0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
            0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
            0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
        );
        i_vol
    }

    /// Compute the deviatoric projection matrix I_dev.
    ///
    /// I_dev = I - I_vol
    ///
    /// This 6×6 matrix projects a strain vector onto its deviatoric component.
    fn deviatoric_projector() -> Matrix6<f64> {
        let i_vol = Self::volumetric_projector();
        Matrix6::identity() - i_vol
    }

    /// Compute the volumetric part of the B-matrix: B_vol = I_vol * B
    fn compute_b_vol(b: &DMatrix<f64>) -> DMatrix<f64> {
        let i_vol = Self::volumetric_projector();
        let i_vol_dm = DMatrix::from_iterator(6, 6, i_vol.iter().cloned());
        i_vol_dm * b
    }

    /// Compute the deviatoric part of the B-matrix: B_dev = I_dev * B
    fn compute_b_dev(b: &DMatrix<f64>) -> DMatrix<f64> {
        let i_dev = Self::deviatoric_projector();
        let i_dev_dm = DMatrix::from_iterator(6, 6, i_dev.iter().cloned());
        i_dev_dm * b
    }

    /// Compute the element-averaged volumetric B-matrix.
    ///
    /// B̄_vol = (1/V) ∫ B_vol dV
    ///
    /// This is computed using 2×2×2 Gauss integration.
    fn compute_bbar_vol(coords: &[Point3]) -> (DMatrix<f64>, f64) {
        let gauss_points = gauss_hex(2);
        let mut bbar_vol = DMatrix::zeros(6, 24);
        let mut total_volume = 0.0;

        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let zeta = gp.zeta();
            let w = gp.weight;

            let (b, det_j) = Self::compute_b_matrix(coords, xi, eta, zeta);
            let b_vol = Self::compute_b_vol(&b);

            // Accumulate weighted B_vol
            bbar_vol += &b_vol * (det_j * w);
            total_volume += det_j * w;
        }

        // Divide by volume to get average
        bbar_vol /= total_volume;

        (bbar_vol, total_volume)
    }

    /// Compute the B-bar matrix at a specific integration point.
    ///
    /// B̄ = B_dev + B̄_vol
    ///
    /// where B_dev is computed locally and B̄_vol is the element-averaged volumetric B.
    fn compute_bbar_at_point(
        coords: &[Point3],
        xi: f64,
        eta: f64,
        zeta: f64,
        bbar_vol: &DMatrix<f64>,
    ) -> (DMatrix<f64>, f64) {
        let (b, det_j) = Self::compute_b_matrix(coords, xi, eta, zeta);
        let b_dev = Self::compute_b_dev(&b);

        // B̄ = B_dev + B̄_vol
        let bbar = b_dev + bbar_vol;

        (bbar, det_j)
    }
}

impl Element for Hex8Bbar {
    fn n_nodes(&self) -> usize {
        8
    }

    fn dofs_per_node(&self) -> usize {
        3
    }

    fn stiffness(&self, coords: &[Point3], material: &Material) -> DMatrix<f64> {
        assert_eq!(
            coords.len(),
            8,
            "Hex8Bbar requires exactly 8 nodal coordinates"
        );

        let d = material.constitutive_3d();
        let mut k = DMatrix::zeros(24, 24);

        // Precompute element-averaged volumetric B-matrix
        let (bbar_vol, _volume) = Self::compute_bbar_vol(coords);

        // Integrate stiffness using B-bar formulation
        let gauss_points = gauss_hex(2);

        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let zeta = gp.zeta();
            let w = gp.weight;

            // Compute B̄ at this integration point
            let (bbar, det_j) = Self::compute_bbar_at_point(coords, xi, eta, zeta, &bbar_vol);

            // K += B̄^T * D * B̄ * |J| * w
            let db = &d * &bbar;
            let btdb = bbar.transpose() * db;
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
            8,
            "Hex8Bbar requires exactly 8 nodal coordinates"
        );
        assert_eq!(
            displacements.len(),
            24,
            "Hex8Bbar requires 24 displacement DOFs"
        );

        let d = material.constitutive_3d();
        let u = nalgebra::DVector::from_row_slice(displacements);

        // Precompute element-averaged volumetric B-matrix
        let (bbar_vol, _volume) = Self::compute_bbar_vol(coords);

        // Compute stress at each integration point using B-bar
        let gauss_points = gauss_hex(2);
        let mut stresses = Vec::with_capacity(gauss_points.len());

        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let zeta = gp.zeta();

            let (bbar, _det_j) = Self::compute_bbar_at_point(coords, xi, eta, zeta, &bbar_vol);

            // ε = B̄ * u
            let strain = &bbar * &u; // 6×1

            // σ = D * ε
            let stress_vec = &d * Vector6::from_iterator(strain.iter().cloned());

            stresses.push(StressTensor(stress_vec));
        }

        stresses
    }

    fn volume(&self, coords: &[Point3]) -> f64 {
        assert_eq!(
            coords.len(),
            8,
            "Hex8Bbar requires exactly 8 nodal coordinates"
        );

        let gauss_points = gauss_hex(2);
        let mut volume = 0.0;

        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let zeta = gp.zeta();
            let w = gp.weight;

            let (dn_dxi, dn_deta, dn_dzeta) = Self::shape_derivatives(xi, eta, zeta);
            let j = Self::jacobian(coords, &dn_dxi, &dn_deta, &dn_dzeta);
            let det_j = j.determinant();

            volume += det_j * w;
        }

        volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Unit cube with vertices at (0,0,0) to (1,1,1).
    fn unit_cube() -> Vec<Point3> {
        vec![
            Point3::new(0.0, 0.0, 0.0), // 0
            Point3::new(1.0, 0.0, 0.0), // 1
            Point3::new(1.0, 1.0, 0.0), // 2
            Point3::new(0.0, 1.0, 0.0), // 3
            Point3::new(0.0, 0.0, 1.0), // 4
            Point3::new(1.0, 0.0, 1.0), // 5
            Point3::new(1.0, 1.0, 1.0), // 6
            Point3::new(0.0, 1.0, 1.0), // 7
        ]
    }

    /// Rectangular prism 2×1×1.
    fn stretched_cube() -> Vec<Point3> {
        vec![
            Point3::new(0.0, 0.0, 0.0), // 0
            Point3::new(2.0, 0.0, 0.0), // 1
            Point3::new(2.0, 1.0, 0.0), // 2
            Point3::new(0.0, 1.0, 0.0), // 3
            Point3::new(0.0, 0.0, 1.0), // 4
            Point3::new(2.0, 0.0, 1.0), // 5
            Point3::new(2.0, 1.0, 1.0), // 6
            Point3::new(0.0, 1.0, 1.0), // 7
        ]
    }

    #[test]
    fn test_hex8bbar_shape_functions_sum_to_one() {
        // Shape functions should sum to 1 at any point
        let test_points = [
            (0.0, 0.0, 0.0),  // center
            (1.0, 1.0, 1.0),  // corner
            (-1.0, 0.5, 0.0), // face
            (0.5, 0.5, 0.5),  // arbitrary
        ];

        for (xi, eta, zeta) in test_points {
            let n = Hex8Bbar::shape_functions(xi, eta, zeta);
            let sum: f64 = n.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_hex8bbar_shape_functions_at_nodes() {
        // N_i = 1 at node i, 0 at other nodes
        for i in 0..8 {
            let n = Hex8Bbar::shape_functions(XI[i], ETA[i], ZETA[i]);
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(n[j], expected, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_hex8bbar_volume_unit_cube() {
        let hex = Hex8Bbar::new();
        let coords = unit_cube();
        let vol = hex.volume(&coords);
        assert_relative_eq!(vol, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_hex8bbar_volume_stretched() {
        let hex = Hex8Bbar::new();
        let coords = stretched_cube();
        let vol = hex.volume(&coords);
        assert_relative_eq!(vol, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_hex8bbar_node_count() {
        let hex = Hex8Bbar::new();
        assert_eq!(hex.n_nodes(), 8);
        assert_eq!(hex.dofs_per_node(), 3);
        assert_eq!(hex.n_dofs(), 24);
    }

    #[test]
    fn test_hex8bbar_stiffness_symmetric() {
        let hex = Hex8Bbar::new();
        let coords = unit_cube();
        let mat = Material::steel();

        let k = hex.stiffness(&coords, &mat);

        assert_eq!(k.nrows(), 24);
        assert_eq!(k.ncols(), 24);

        // Use relative tolerance for large values
        for i in 0..24 {
            for j in 0..24 {
                assert_relative_eq!(k[(i, j)], k[(j, i)], max_relative = 1e-12);
            }
        }
    }

    #[test]
    fn test_hex8bbar_stiffness_positive_diagonal() {
        let hex = Hex8Bbar::new();
        let coords = unit_cube();
        let mat = Material::steel();

        let k = hex.stiffness(&coords, &mat);

        for i in 0..24 {
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
    fn test_hex8bbar_rigid_body_translation() {
        let hex = Hex8Bbar::new();
        let coords = unit_cube();
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = hex.stiffness(&coords, &mat);

        // x-translation: all nodes move by (1, 0, 0)
        let u_x = nalgebra::DVector::from_vec(vec![
            1.0, 0.0, 0.0, // node 0
            1.0, 0.0, 0.0, // node 1
            1.0, 0.0, 0.0, // node 2
            1.0, 0.0, 0.0, // node 3
            1.0, 0.0, 0.0, // node 4
            1.0, 0.0, 0.0, // node 5
            1.0, 0.0, 0.0, // node 6
            1.0, 0.0, 0.0, // node 7
        ]);
        let f_x = &k * &u_x;
        assert_relative_eq!(f_x.norm(), 0.0, epsilon = 1e-10);

        // y-translation
        let u_y = nalgebra::DVector::from_vec(vec![
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ]);
        let f_y = &k * &u_y;
        assert_relative_eq!(f_y.norm(), 0.0, epsilon = 1e-10);

        // z-translation
        let u_z = nalgebra::DVector::from_vec(vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ]);
        let f_z = &k * &u_z;
        assert_relative_eq!(f_z.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hex8bbar_constant_strain_patch() {
        // Patch test: impose uniform ε_xx = 0.001
        // Displacement field: u = 0.001 * x, v = 0, w = 0
        let hex = Hex8Bbar::new();
        let coords = unit_cube();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Displacements: u = 0.001 * x for each node
        let strain_xx = 0.001;
        let displacements = [
            0.0 * strain_xx,
            0.0,
            0.0, // node 0: x=0
            1.0 * strain_xx,
            0.0,
            0.0, // node 1: x=1
            1.0 * strain_xx,
            0.0,
            0.0, // node 2: x=1
            0.0 * strain_xx,
            0.0,
            0.0, // node 3: x=0
            0.0 * strain_xx,
            0.0,
            0.0, // node 4: x=0
            1.0 * strain_xx,
            0.0,
            0.0, // node 5: x=1
            1.0 * strain_xx,
            0.0,
            0.0, // node 6: x=1
            0.0 * strain_xx,
            0.0,
            0.0, // node 7: x=0
        ];

        let stresses = hex.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 8); // 2×2×2 integration points

        // All integration points should have the same stress (constant strain)
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
    fn test_hex8bbar_stress_count() {
        let hex = Hex8Bbar::new();
        let coords = unit_cube();
        let mat = Material::steel();
        let displacements = [0.0; 24];

        let stresses = hex.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 8); // One stress per integration point
    }

    #[test]
    fn test_volumetric_projector() {
        let i_vol = Hex8Bbar::volumetric_projector();

        // Test that I_vol is idempotent: I_vol * I_vol = I_vol
        let i_vol_sq = i_vol * i_vol;
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(i_vol_sq[(i, j)], i_vol[(i, j)], epsilon = 1e-14);
            }
        }

        // Test that I_vol projects [1, 1, 1, 0, 0, 0]^T to itself
        let m = Vector6::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        let projected = i_vol * m;
        assert_relative_eq!(projected[0], 1.0, epsilon = 1e-14);
        assert_relative_eq!(projected[1], 1.0, epsilon = 1e-14);
        assert_relative_eq!(projected[2], 1.0, epsilon = 1e-14);
        assert_relative_eq!(projected[3], 0.0, epsilon = 1e-14);
        assert_relative_eq!(projected[4], 0.0, epsilon = 1e-14);
        assert_relative_eq!(projected[5], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_deviatoric_projector() {
        let i_dev = Hex8Bbar::deviatoric_projector();

        // Test that I_dev is idempotent: I_dev * I_dev = I_dev
        let i_dev_sq = i_dev * i_dev;
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(i_dev_sq[(i, j)], i_dev[(i, j)], epsilon = 1e-14);
            }
        }

        // Test that I_dev projects [1, 1, 1, 0, 0, 0]^T to zero
        let m = Vector6::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        let projected = i_dev * m;
        for i in 0..6 {
            assert_relative_eq!(projected[i], 0.0, epsilon = 1e-14);
        }

        // Test that I_vol + I_dev = I
        let i_vol = Hex8Bbar::volumetric_projector();
        let sum = i_vol + i_dev;
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(sum[(i, j)], expected, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_hex8bbar_no_hourglass_modes() {
        // Test that the B-bar element does not have zero-energy hourglass modes.
        // Apply an hourglass mode deformation and verify non-zero strain energy.
        let hex = Hex8Bbar::new();
        let coords = unit_cube();
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = hex.stiffness(&coords, &mat);

        // Classic hourglass mode for hex8: alternating +/- displacement
        // This is the characteristic mode that reduced integration can't detect
        let hourglass = nalgebra::DVector::from_vec(vec![
            1.0, 0.0, 0.0, // node 0: +x
            -1.0, 0.0, 0.0, // node 1: -x
            1.0, 0.0, 0.0, // node 2: +x
            -1.0, 0.0, 0.0, // node 3: -x
            -1.0, 0.0, 0.0, // node 4: -x
            1.0, 0.0, 0.0, // node 5: +x
            -1.0, 0.0, 0.0, // node 6: -x
            1.0, 0.0, 0.0, // node 7: +x
        ]);

        // Strain energy = 0.5 * u^T * K * u
        let ku = &k * &hourglass;
        let energy = 0.5 * hourglass.dot(&ku);

        // Energy should be positive (no zero-energy mode)
        assert!(
            energy > 1e-10,
            "Hourglass mode has near-zero energy ({}), indicating rank deficiency",
            energy
        );
    }

    #[test]
    fn test_hex8bbar_stiffness_differs_from_hex8() {
        // The B-bar element should produce a different stiffness matrix
        // than the standard Hex8 element.
        use crate::element::Hex8;

        let coords = unit_cube();
        let mat = Material::steel();

        let hex8 = Hex8::new();
        let hex8_bbar = Hex8Bbar::new();

        let k_std = hex8.stiffness(&coords, &mat);
        let k_bbar = hex8_bbar.stiffness(&coords, &mat);

        // The matrices should be different
        let mut diff_found = false;
        for i in 0..24 {
            for j in 0..24 {
                if (k_std[(i, j)] - k_bbar[(i, j)]).abs() > 1e-6 {
                    diff_found = true;
                    break;
                }
            }
            if diff_found {
                break;
            }
        }

        assert!(
            diff_found,
            "Hex8Bbar stiffness should differ from standard Hex8"
        );
    }

    #[test]
    fn test_hex8bbar_nearly_incompressible() {
        // Test that B-bar handles nearly incompressible material (nu → 0.5)
        // Standard Hex8 would have numerical issues with high nu
        let hex = Hex8Bbar::new();
        let coords = unit_cube();
        let mat = Material::new(1e6, 0.499).unwrap(); // Nearly incompressible

        let k = hex.stiffness(&coords, &mat);

        // Verify stiffness matrix is valid
        assert_eq!(k.nrows(), 24);
        assert_eq!(k.ncols(), 24);

        // Should be symmetric
        for i in 0..24 {
            for j in 0..24 {
                assert_relative_eq!(k[(i, j)], k[(j, i)], max_relative = 1e-10);
            }
        }

        // Diagonal should be positive
        for i in 0..24 {
            assert!(k[(i, i)] > 0.0, "K[{},{}] should be positive", i, i);
        }

        // Rigid body modes should still have zero strain energy
        let u_x = nalgebra::DVector::from_vec(vec![
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ]);
        let f_x = &k * &u_x;
        assert_relative_eq!(f_x.norm(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_hex8bbar_stiffness_differs_from_hex8_sri() {
        // B-bar and SRI use different locking mitigation strategies,
        // so their stiffness matrices should be different.
        use crate::element::Hex8SRI;

        let coords = unit_cube();
        let mat = Material::steel();

        let hex8_sri = Hex8SRI::new();
        let hex8_bbar = Hex8Bbar::new();

        let k_sri = hex8_sri.stiffness(&coords, &mat);
        let k_bbar = hex8_bbar.stiffness(&coords, &mat);

        // The matrices should be different
        let mut diff_found = false;
        for i in 0..24 {
            for j in 0..24 {
                if (k_sri[(i, j)] - k_bbar[(i, j)]).abs() > 1e-6 {
                    diff_found = true;
                    break;
                }
            }
            if diff_found {
                break;
            }
        }

        assert!(
            diff_found,
            "Hex8Bbar stiffness should differ from Hex8SRI"
        );
    }

    #[test]
    fn test_hex8bbar_hydrostatic_compression() {
        // Apply hydrostatic compression and verify consistent volumetric response.
        // This tests that the averaged volumetric strain works correctly.
        let hex = Hex8Bbar::new();
        let coords = unit_cube();
        let mat = Material::new(1e6, 0.3).unwrap();

        // Uniform compression: u = -ε*x, v = -ε*y, w = -ε*z
        let strain = 0.001;
        let displacements = [
            -strain * 0.0,
            -strain * 0.0,
            -strain * 0.0, // node 0: (0,0,0)
            -strain * 1.0,
            -strain * 0.0,
            -strain * 0.0, // node 1: (1,0,0)
            -strain * 1.0,
            -strain * 1.0,
            -strain * 0.0, // node 2: (1,1,0)
            -strain * 0.0,
            -strain * 1.0,
            -strain * 0.0, // node 3: (0,1,0)
            -strain * 0.0,
            -strain * 0.0,
            -strain * 1.0, // node 4: (0,0,1)
            -strain * 1.0,
            -strain * 0.0,
            -strain * 1.0, // node 5: (1,0,1)
            -strain * 1.0,
            -strain * 1.0,
            -strain * 1.0, // node 6: (1,1,1)
            -strain * 0.0,
            -strain * 1.0,
            -strain * 1.0, // node 7: (0,1,1)
        ];

        let stresses = hex.stress(&coords, &displacements, &mat);

        // All integration points should have the same stress (uniform volumetric strain)
        let first_stress = &stresses[0].0;
        for stress in &stresses[1..] {
            for i in 0..6 {
                assert_relative_eq!(stress.0[i], first_stress[i], epsilon = 1e-6);
            }
        }

        // Principal stresses should be equal (hydrostatic)
        assert_relative_eq!(first_stress[0], first_stress[1], epsilon = 1e-6);
        assert_relative_eq!(first_stress[1], first_stress[2], epsilon = 1e-6);

        // Shear stresses should be zero
        assert_relative_eq!(first_stress[3], 0.0, epsilon = 1e-10);
        assert_relative_eq!(first_stress[4], 0.0, epsilon = 1e-10);
        assert_relative_eq!(first_stress[5], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hex8bbar_b_matrix_decomposition() {
        // Verify B_vol + B_dev = B
        let coords = unit_cube();

        // Get B-matrix at center
        let (b, _det_j) = Hex8Bbar::compute_b_matrix(&coords, 0.0, 0.0, 0.0);
        let b_vol = Hex8Bbar::compute_b_vol(&b);
        let b_dev = Hex8Bbar::compute_b_dev(&b);

        // B_vol + B_dev should equal B
        let b_sum = &b_vol + &b_dev;
        for i in 0..6 {
            for j in 0..24 {
                assert_relative_eq!(b_sum[(i, j)], b[(i, j)], epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_hex8bbar_volumetric_averaging() {
        // Verify that the averaged volumetric B-matrix is computed correctly.
        // For a unit cube, the averaged B_vol should have non-zero entries
        // only in the first 3 rows (volumetric strain components).
        let coords = unit_cube();

        let (bbar_vol, volume) = Hex8Bbar::compute_bbar_vol(&coords);

        // Volume should be 1.0 for unit cube
        assert_relative_eq!(volume, 1.0, epsilon = 1e-12);

        // The averaged B_vol should have zero entries in shear rows (3-5)
        for i in 3..6 {
            for j in 0..24 {
                assert_relative_eq!(
                    bbar_vol[(i, j)],
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14
                );
            }
        }
    }
}
