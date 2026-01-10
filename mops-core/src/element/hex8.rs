//! 8-node hexahedron (Hex8) element.
//!
//! The Hex8 is a standard 3D solid element with:
//! - 8 nodes at vertices of a hexahedron
//! - 3 DOFs per node (u, v, w displacements)
//! - 24 total DOFs
//! - Trilinear shape functions
//! - 2×2×2 Gauss integration (8 points)
//!
//! # Shape Functions
//!
//! Trilinear shape functions in natural coordinates (ξ, η, ζ) ∈ [-1, 1]³:
//! ```text
//! N_i = (1 + ξ_i*ξ)(1 + η_i*η)(1 + ζ_i*ζ) / 8
//! ```
//! where (ξ_i, η_i, ζ_i) are ±1 for node i.
//!
//! # Node Numbering
//!
//! Standard node numbering (counterclockwise when viewed from outside):
//! ```text
//!        7-------6
//!       /|      /|
//!      / |     / |
//!     4-------5  |
//!     |  3----|--2
//!     | /     | /
//!     |/      |/
//!     0-------1
//!
//! Node 0: (-1, -1, -1)
//! Node 1: (+1, -1, -1)
//! Node 2: (+1, +1, -1)
//! Node 3: (-1, +1, -1)
//! Node 4: (-1, -1, +1)
//! Node 5: (+1, -1, +1)
//! Node 6: (+1, +1, +1)
//! Node 7: (-1, +1, +1)
//! ```
//!
//! # Limitations
//!
//! - Shear locking in bending-dominated problems
//! - Best accuracy when element is close to rectangular
//! - Distorted elements reduce accuracy

use crate::element::gauss::gauss_hex;
use crate::element::Element;
use crate::material::Material;
use crate::types::{Point3, StressTensor};
use nalgebra::{DMatrix, Matrix3, Vector3, Vector6};

/// Natural coordinates for each of the 8 nodes.
/// Node i has natural coordinates (XI[i], ETA[i], ZETA[i]).
const XI: [f64; 8] = [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0];
const ETA: [f64; 8] = [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0];
const ZETA: [f64; 8] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];

/// 8-node hexahedral element (trilinear brick).
#[derive(Debug, Clone, Copy, Default)]
pub struct Hex8;

impl Hex8 {
    /// Create a new Hex8 element.
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
        // Actually: [dN/dx, dN/dy, dN/dz]^T = J^(-1) * [dN/dξ, dN/dη, dN/dζ]^T
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
}

impl Element for Hex8 {
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
            "Hex8 requires exactly 8 nodal coordinates"
        );

        let d = material.constitutive_3d();
        let mut k = DMatrix::zeros(24, 24);

        // 2×2×2 Gauss integration
        let gauss_points = gauss_hex(2);

        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let zeta = gp.zeta();
            let w = gp.weight;

            let (b, det_j) = Self::compute_b_matrix(coords, xi, eta, zeta);

            // K += B^T * D * B * |J| * w
            let db = &d * &b; // 6×24
            let btdb = b.transpose() * db; // 24×24

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
            "Hex8 requires exactly 8 nodal coordinates"
        );
        assert_eq!(
            displacements.len(),
            24,
            "Hex8 requires 24 displacement DOFs"
        );

        let d = material.constitutive_3d();
        let u = nalgebra::DVector::from_row_slice(displacements);

        // Compute stress at each integration point
        let gauss_points = gauss_hex(2);
        let mut stresses = Vec::with_capacity(gauss_points.len());

        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let zeta = gp.zeta();

            let (b, _det_j) = Self::compute_b_matrix(coords, xi, eta, zeta);

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
            8,
            "Hex8 requires exactly 8 nodal coordinates"
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
    fn test_hex8_shape_functions_sum_to_one() {
        // Shape functions should sum to 1 at any point
        let test_points = [
            (0.0, 0.0, 0.0),   // center
            (1.0, 1.0, 1.0),   // corner
            (-1.0, 0.5, 0.0),  // face
            (0.5, 0.5, 0.5),   // arbitrary
        ];

        for (xi, eta, zeta) in test_points {
            let n = Hex8::shape_functions(xi, eta, zeta);
            let sum: f64 = n.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_hex8_shape_functions_at_nodes() {
        // N_i = 1 at node i, 0 at other nodes
        for i in 0..8 {
            let n = Hex8::shape_functions(XI[i], ETA[i], ZETA[i]);
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(n[j], expected, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_hex8_volume_unit_cube() {
        let hex = Hex8::new();
        let coords = unit_cube();
        let vol = hex.volume(&coords);
        assert_relative_eq!(vol, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_hex8_volume_stretched() {
        let hex = Hex8::new();
        let coords = stretched_cube();
        let vol = hex.volume(&coords);
        assert_relative_eq!(vol, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_hex8_node_count() {
        let hex = Hex8::new();
        assert_eq!(hex.n_nodes(), 8);
        assert_eq!(hex.dofs_per_node(), 3);
        assert_eq!(hex.n_dofs(), 24);
    }

    #[test]
    fn test_hex8_stiffness_symmetric() {
        let hex = Hex8::new();
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
    fn test_hex8_stiffness_positive_diagonal() {
        let hex = Hex8::new();
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
    fn test_hex8_rigid_body_translation() {
        let hex = Hex8::new();
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
    fn test_hex8_constant_strain_patch() {
        // Patch test: impose uniform ε_xx = 0.001
        // Displacement field: u = 0.001 * x, v = 0, w = 0
        let hex = Hex8::new();
        let coords = unit_cube();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Displacements: u = 0.001 * x for each node
        let strain_xx = 0.001;
        let displacements = [
            0.0 * strain_xx, 0.0, 0.0,  // node 0: x=0
            1.0 * strain_xx, 0.0, 0.0,  // node 1: x=1
            1.0 * strain_xx, 0.0, 0.0,  // node 2: x=1
            0.0 * strain_xx, 0.0, 0.0,  // node 3: x=0
            0.0 * strain_xx, 0.0, 0.0,  // node 4: x=0
            1.0 * strain_xx, 0.0, 0.0,  // node 5: x=1
            1.0 * strain_xx, 0.0, 0.0,  // node 6: x=1
            0.0 * strain_xx, 0.0, 0.0,  // node 7: x=0
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
    fn test_hex8_stress_count() {
        let hex = Hex8::new();
        let coords = unit_cube();
        let mat = Material::steel();
        let displacements = [0.0; 24];

        let stresses = hex.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 8); // One stress per integration point
    }
}
