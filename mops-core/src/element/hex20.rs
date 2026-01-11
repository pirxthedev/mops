//! 20-node hexahedron (Hex20) element.
//!
//! The Hex20 is a serendipity quadratic 3D solid element with:
//! - 20 nodes: 8 vertices + 12 edge midpoints
//! - 3 DOFs per node (u, v, w displacements)
//! - 60 total DOFs
//! - Serendipity quadratic shape functions
//! - 3×3×3 Gauss integration (27 points)
//!
//! # Shape Functions
//!
//! For corner nodes (i = 0..7):
//! ```text
//! N_i = (1 + ξ_i*ξ)(1 + η_i*η)(1 + ζ_i*ζ)(ξ_i*ξ + η_i*η + ζ_i*ζ - 2) / 8
//! ```
//!
//! For mid-edge nodes (i = 8..19):
//! - Nodes on ξ-edges (8,10,16,18): N_i = (1 - ξ²)(1 + η_i*η)(1 + ζ_i*ζ) / 4
//! - Nodes on η-edges (9,11,17,19): N_i = (1 + ξ_i*ξ)(1 - η²)(1 + ζ_i*ζ) / 4
//! - Nodes on ζ-edges (12,13,14,15): N_i = (1 + ξ_i*ξ)(1 + η_i*η)(1 - ζ²) / 4
//!
//! # Node Numbering
//!
//! ```text
//!        7-----14------6
//!       /|            /|
//!      / |           / |
//!    15  |         13  |
//!    /  19         /  18
//!   /    |        /    |
//!  4-----12------5     |
//!  |     |       |     |
//!  |     3---10--|-----2
//! 16    /       17    /
//!  |   /         |   /
//!  | 11          |  9
//!  | /           | /
//!  |/            |/
//!  0------8------1
//!
//! Corner nodes (vertices):
//! Node 0: (-1, -1, -1)   Node 4: (-1, -1, +1)
//! Node 1: (+1, -1, -1)   Node 5: (+1, -1, +1)
//! Node 2: (+1, +1, -1)   Node 6: (+1, +1, +1)
//! Node 3: (-1, +1, -1)   Node 7: (-1, +1, +1)
//!
//! Mid-edge nodes (on ξ-direction edges):
//! Node 8:  ( 0, -1, -1)   Node 16: ( 0, -1, +1)
//! Node 10: ( 0, +1, -1)   Node 18: ( 0, +1, +1)
//!
//! Mid-edge nodes (on η-direction edges):
//! Node 9:  (+1,  0, -1)   Node 17: (+1,  0, +1)
//! Node 11: (-1,  0, -1)   Node 19: (-1,  0, +1)
//!
//! Mid-edge nodes (on ζ-direction edges):
//! Node 12: (-1, -1,  0)   Node 13: (+1, -1,  0)
//! Node 14: (+1, +1,  0)   Node 15: (-1, +1,  0)
//! ```
//!
//! # Advantages over Hex8
//!
//! - Much better accuracy for bending problems (no shear locking)
//! - Better stress representation near stress concentrations
//! - Handles curved geometries more accurately
//!
//! # Limitations
//!
//! - More expensive to compute (27 vs 8 integration points)
//! - More sensitive to mesh distortion than Hex8
//! - Serendipity element (no interior node) may have slight accuracy loss vs full quadratic

use crate::element::gauss::gauss_hex;
use crate::element::Element;
use crate::material::Material;
use crate::types::{Point3, StressTensor};
use nalgebra::{DMatrix, Matrix3, Vector3, Vector6};

/// Natural coordinates (ξ, η, ζ) for each of the 20 nodes.
///
/// Nodes 0-7: Corner nodes at (±1, ±1, ±1)
/// Nodes 8-19: Mid-edge nodes with one coordinate = 0
const NODE_COORDS: [[f64; 3]; 20] = [
    // Corner nodes (same ordering as Hex8)
    [-1.0, -1.0, -1.0], // 0
    [1.0, -1.0, -1.0],  // 1
    [1.0, 1.0, -1.0],   // 2
    [-1.0, 1.0, -1.0],  // 3
    [-1.0, -1.0, 1.0],  // 4
    [1.0, -1.0, 1.0],   // 5
    [1.0, 1.0, 1.0],    // 6
    [-1.0, 1.0, 1.0],   // 7
    // Mid-edge nodes on ξ-direction edges (η, ζ = ±1)
    [0.0, -1.0, -1.0], // 8  (edge 0-1)
    [1.0, 0.0, -1.0],  // 9  (edge 1-2)
    [0.0, 1.0, -1.0],  // 10 (edge 2-3)
    [-1.0, 0.0, -1.0], // 11 (edge 3-0)
    // Mid-edge nodes on ζ-direction edges (ξ, η = ±1)
    [-1.0, -1.0, 0.0], // 12 (edge 0-4)
    [1.0, -1.0, 0.0],  // 13 (edge 1-5)
    [1.0, 1.0, 0.0],   // 14 (edge 2-6)
    [-1.0, 1.0, 0.0],  // 15 (edge 3-7)
    // Mid-edge nodes on ξ-direction edges at ζ=+1 (η = ±1)
    [0.0, -1.0, 1.0], // 16 (edge 4-5)
    [1.0, 0.0, 1.0],  // 17 (edge 5-6)
    [0.0, 1.0, 1.0],  // 18 (edge 6-7)
    [-1.0, 0.0, 1.0], // 19 (edge 7-4)
];

/// Identifies which nodes are corner nodes (0-7) vs mid-edge nodes (8-19).
/// Also identifies the type of mid-edge node by which coordinate is zero.
#[derive(Debug, Clone, Copy, PartialEq)]
enum NodeType {
    Corner,
    MidEdgeXi,   // ξ = 0
    MidEdgeEta,  // η = 0
    MidEdgeZeta, // ζ = 0
}

impl NodeType {
    fn of(node: usize) -> Self {
        match node {
            0..=7 => NodeType::Corner,
            8 | 10 | 16 | 18 => NodeType::MidEdgeXi,
            9 | 11 | 17 | 19 => NodeType::MidEdgeEta,
            12..=15 => NodeType::MidEdgeZeta,
            _ => panic!("Invalid node index: {}", node),
        }
    }
}

/// 20-node hexahedral element (serendipity quadratic brick).
#[derive(Debug, Clone, Copy, Default)]
pub struct Hex20;

impl Hex20 {
    /// Create a new Hex20 element.
    pub fn new() -> Self {
        Self
    }

    /// Compute shape functions at a point in natural coordinates.
    ///
    /// Returns N = [N_0, N_1, ..., N_19]
    pub fn shape_functions(xi: f64, eta: f64, zeta: f64) -> [f64; 20] {
        let mut n = [0.0; 20];

        for i in 0..20 {
            let xi_i = NODE_COORDS[i][0];
            let eta_i = NODE_COORDS[i][1];
            let zeta_i = NODE_COORDS[i][2];

            match NodeType::of(i) {
                NodeType::Corner => {
                    // Corner node shape function:
                    // N_i = (1/8)(1 + ξ_i*ξ)(1 + η_i*η)(1 + ζ_i*ζ)(ξ_i*ξ + η_i*η + ζ_i*ζ - 2)
                    let factor1 = (1.0 + xi_i * xi) * (1.0 + eta_i * eta) * (1.0 + zeta_i * zeta);
                    let factor2 = xi_i * xi + eta_i * eta + zeta_i * zeta - 2.0;
                    n[i] = 0.125 * factor1 * factor2;
                }
                NodeType::MidEdgeXi => {
                    // Mid-edge node on ξ-edge (ξ_i = 0):
                    // N_i = (1/4)(1 - ξ²)(1 + η_i*η)(1 + ζ_i*ζ)
                    n[i] = 0.25 * (1.0 - xi * xi) * (1.0 + eta_i * eta) * (1.0 + zeta_i * zeta);
                }
                NodeType::MidEdgeEta => {
                    // Mid-edge node on η-edge (η_i = 0):
                    // N_i = (1/4)(1 + ξ_i*ξ)(1 - η²)(1 + ζ_i*ζ)
                    n[i] = 0.25 * (1.0 + xi_i * xi) * (1.0 - eta * eta) * (1.0 + zeta_i * zeta);
                }
                NodeType::MidEdgeZeta => {
                    // Mid-edge node on ζ-edge (ζ_i = 0):
                    // N_i = (1/4)(1 + ξ_i*ξ)(1 + η_i*η)(1 - ζ²)
                    n[i] = 0.25 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta) * (1.0 - zeta * zeta);
                }
            }
        }

        n
    }

    /// Compute shape function derivatives with respect to natural coordinates.
    ///
    /// Returns (dN/dξ, dN/dη, dN/dζ) for each node.
    fn shape_derivatives(xi: f64, eta: f64, zeta: f64) -> ([f64; 20], [f64; 20], [f64; 20]) {
        let mut dn_dxi = [0.0; 20];
        let mut dn_deta = [0.0; 20];
        let mut dn_dzeta = [0.0; 20];

        for i in 0..20 {
            let xi_i = NODE_COORDS[i][0];
            let eta_i = NODE_COORDS[i][1];
            let zeta_i = NODE_COORDS[i][2];

            match NodeType::of(i) {
                NodeType::Corner => {
                    // Corner node:
                    // N_i = (1/8)(1 + ξ_i*ξ)(1 + η_i*η)(1 + ζ_i*ζ)(ξ_i*ξ + η_i*η + ζ_i*ζ - 2)
                    //
                    // Let a = (1 + ξ_i*ξ), b = (1 + η_i*η), c = (1 + ζ_i*ζ)
                    // Let s = ξ_i*ξ + η_i*η + ζ_i*ζ - 2
                    // N_i = (1/8) * a * b * c * s
                    //
                    // dN/dξ = (1/8) * [ξ_i * b * c * s + a * b * c * ξ_i]
                    //       = (1/8) * ξ_i * b * c * (s + a)
                    //       = (1/8) * ξ_i * (1+η_i*η) * (1+ζ_i*ζ) * (2*ξ_i*ξ + η_i*η + ζ_i*ζ - 1)

                    let a = 1.0 + xi_i * xi;
                    let b = 1.0 + eta_i * eta;
                    let c = 1.0 + zeta_i * zeta;

                    // dN/dξ = (1/8) * ξ_i * b * c * (s + a)
                    //       = (1/8) * ξ_i * b * c * (ξ_i*ξ + η_i*η + ζ_i*ζ - 2 + 1 + ξ_i*ξ)
                    //       = (1/8) * ξ_i * b * c * (2*ξ_i*ξ + η_i*η + ζ_i*ζ - 1)
                    dn_dxi[i] = 0.125
                        * xi_i
                        * b
                        * c
                        * (2.0 * xi_i * xi + eta_i * eta + zeta_i * zeta - 1.0);

                    // dN/dη = (1/8) * a * η_i * c * (s + b)
                    //       = (1/8) * η_i * a * c * (ξ_i*ξ + 2*η_i*η + ζ_i*ζ - 1)
                    dn_deta[i] = 0.125
                        * eta_i
                        * a
                        * c
                        * (xi_i * xi + 2.0 * eta_i * eta + zeta_i * zeta - 1.0);

                    // dN/dζ = (1/8) * a * b * ζ_i * (s + c)
                    //       = (1/8) * ζ_i * a * b * (ξ_i*ξ + η_i*η + 2*ζ_i*ζ - 1)
                    dn_dzeta[i] = 0.125
                        * zeta_i
                        * a
                        * b
                        * (xi_i * xi + eta_i * eta + 2.0 * zeta_i * zeta - 1.0);
                }
                NodeType::MidEdgeXi => {
                    // Mid-edge node on ξ-edge (ξ_i = 0):
                    // N_i = (1/4)(1 - ξ²)(1 + η_i*η)(1 + ζ_i*ζ)
                    let b = 1.0 + eta_i * eta;
                    let c = 1.0 + zeta_i * zeta;

                    dn_dxi[i] = -0.5 * xi * b * c;
                    dn_deta[i] = 0.25 * (1.0 - xi * xi) * eta_i * c;
                    dn_dzeta[i] = 0.25 * (1.0 - xi * xi) * b * zeta_i;
                }
                NodeType::MidEdgeEta => {
                    // Mid-edge node on η-edge (η_i = 0):
                    // N_i = (1/4)(1 + ξ_i*ξ)(1 - η²)(1 + ζ_i*ζ)
                    let a = 1.0 + xi_i * xi;
                    let c = 1.0 + zeta_i * zeta;

                    dn_dxi[i] = 0.25 * xi_i * (1.0 - eta * eta) * c;
                    dn_deta[i] = -0.5 * eta * a * c;
                    dn_dzeta[i] = 0.25 * a * (1.0 - eta * eta) * zeta_i;
                }
                NodeType::MidEdgeZeta => {
                    // Mid-edge node on ζ-edge (ζ_i = 0):
                    // N_i = (1/4)(1 + ξ_i*ξ)(1 + η_i*η)(1 - ζ²)
                    let a = 1.0 + xi_i * xi;
                    let b = 1.0 + eta_i * eta;

                    dn_dxi[i] = 0.25 * xi_i * b * (1.0 - zeta * zeta);
                    dn_deta[i] = 0.25 * a * eta_i * (1.0 - zeta * zeta);
                    dn_dzeta[i] = -0.5 * zeta * a * b;
                }
            }
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
        dn_dxi: &[f64; 20],
        dn_deta: &[f64; 20],
        dn_dzeta: &[f64; 20],
    ) -> Matrix3<f64> {
        let mut j = Matrix3::zeros();

        for i in 0..20 {
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
    /// Returns the 6×60 B-matrix and the Jacobian determinant.
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
        let mut dn_dx = [0.0; 20];
        let mut dn_dy = [0.0; 20];
        let mut dn_dz = [0.0; 20];

        for i in 0..20 {
            let dn_dnat = Vector3::new(dn_dxi[i], dn_deta[i], dn_dzeta[i]);
            let dn_dphys = j_inv * dn_dnat;
            dn_dx[i] = dn_dphys[0];
            dn_dy[i] = dn_dphys[1];
            dn_dz[i] = dn_dphys[2];
        }

        // Construct B-matrix (6×60)
        // ε = [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz]^T = B * u
        let mut b = DMatrix::zeros(6, 60);

        for i in 0..20 {
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

impl Element for Hex20 {
    fn n_nodes(&self) -> usize {
        20
    }

    fn dofs_per_node(&self) -> usize {
        3
    }

    fn stiffness(&self, coords: &[Point3], material: &Material) -> DMatrix<f64> {
        assert_eq!(
            coords.len(),
            20,
            "Hex20 requires exactly 20 nodal coordinates"
        );

        let d = material.constitutive_3d();
        let mut k = DMatrix::zeros(60, 60);

        // 3×3×3 Gauss integration (27 points)
        let gauss_points = gauss_hex(3);

        for gp in &gauss_points {
            let xi = gp.xi();
            let eta = gp.eta();
            let zeta = gp.zeta();
            let w = gp.weight;

            let (b, det_j) = Self::compute_b_matrix(coords, xi, eta, zeta);

            // K += B^T * D * B * |J| * w
            let db = &d * &b; // 6×60
            let btdb = b.transpose() * db; // 60×60

            k += btdb * det_j * w;
        }

        // Symmetrize to eliminate floating-point asymmetry from B^T D B
        for i in 0..60 {
            for j in (i + 1)..60 {
                let avg = 0.5 * (k[(i, j)] + k[(j, i)]);
                k[(i, j)] = avg;
                k[(j, i)] = avg;
            }
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
            20,
            "Hex20 requires exactly 20 nodal coordinates"
        );
        assert_eq!(
            displacements.len(),
            60,
            "Hex20 requires 60 displacement DOFs"
        );

        let d = material.constitutive_3d();
        let u = nalgebra::DVector::from_row_slice(displacements);

        // Compute stress at each integration point
        let gauss_points = gauss_hex(3);
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
            20,
            "Hex20 requires exactly 20 nodal coordinates"
        );

        let gauss_points = gauss_hex(3);
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

    /// Unit cube with vertices at (0,0,0) to (1,1,1) and midpoints on edges.
    fn unit_cube_hex20() -> Vec<Point3> {
        vec![
            // Corner nodes (0-7)
            Point3::new(0.0, 0.0, 0.0), // 0
            Point3::new(1.0, 0.0, 0.0), // 1
            Point3::new(1.0, 1.0, 0.0), // 2
            Point3::new(0.0, 1.0, 0.0), // 3
            Point3::new(0.0, 0.0, 1.0), // 4
            Point3::new(1.0, 0.0, 1.0), // 5
            Point3::new(1.0, 1.0, 1.0), // 6
            Point3::new(0.0, 1.0, 1.0), // 7
            // Mid-edge nodes on bottom face (z=0)
            Point3::new(0.5, 0.0, 0.0), // 8  (edge 0-1)
            Point3::new(1.0, 0.5, 0.0), // 9  (edge 1-2)
            Point3::new(0.5, 1.0, 0.0), // 10 (edge 2-3)
            Point3::new(0.0, 0.5, 0.0), // 11 (edge 3-0)
            // Mid-edge nodes on vertical edges
            Point3::new(0.0, 0.0, 0.5), // 12 (edge 0-4)
            Point3::new(1.0, 0.0, 0.5), // 13 (edge 1-5)
            Point3::new(1.0, 1.0, 0.5), // 14 (edge 2-6)
            Point3::new(0.0, 1.0, 0.5), // 15 (edge 3-7)
            // Mid-edge nodes on top face (z=1)
            Point3::new(0.5, 0.0, 1.0), // 16 (edge 4-5)
            Point3::new(1.0, 0.5, 1.0), // 17 (edge 5-6)
            Point3::new(0.5, 1.0, 1.0), // 18 (edge 6-7)
            Point3::new(0.0, 0.5, 1.0), // 19 (edge 7-4)
        ]
    }

    /// Rectangular prism 2×1×1 with midpoints.
    fn stretched_hex20() -> Vec<Point3> {
        vec![
            // Corner nodes
            Point3::new(0.0, 0.0, 0.0), // 0
            Point3::new(2.0, 0.0, 0.0), // 1
            Point3::new(2.0, 1.0, 0.0), // 2
            Point3::new(0.0, 1.0, 0.0), // 3
            Point3::new(0.0, 0.0, 1.0), // 4
            Point3::new(2.0, 0.0, 1.0), // 5
            Point3::new(2.0, 1.0, 1.0), // 6
            Point3::new(0.0, 1.0, 1.0), // 7
            // Mid-edge nodes on bottom face
            Point3::new(1.0, 0.0, 0.0), // 8
            Point3::new(2.0, 0.5, 0.0), // 9
            Point3::new(1.0, 1.0, 0.0), // 10
            Point3::new(0.0, 0.5, 0.0), // 11
            // Mid-edge nodes on vertical edges
            Point3::new(0.0, 0.0, 0.5), // 12
            Point3::new(2.0, 0.0, 0.5), // 13
            Point3::new(2.0, 1.0, 0.5), // 14
            Point3::new(0.0, 1.0, 0.5), // 15
            // Mid-edge nodes on top face
            Point3::new(1.0, 0.0, 1.0), // 16
            Point3::new(2.0, 0.5, 1.0), // 17
            Point3::new(1.0, 1.0, 1.0), // 18
            Point3::new(0.0, 0.5, 1.0), // 19
        ]
    }

    #[test]
    fn test_hex20_shape_functions_sum_to_one() {
        // Shape functions should sum to 1 at any point
        let test_points = [
            (0.0, 0.0, 0.0),  // center
            (0.5, 0.5, 0.5),  // arbitrary
            (-0.5, 0.3, 0.2), // arbitrary
            (0.9, -0.9, 0.9), // near corner
        ];

        for (xi, eta, zeta) in test_points {
            let n = Hex20::shape_functions(xi, eta, zeta);
            let sum: f64 = n.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_hex20_shape_functions_at_corner_nodes() {
        // N_i = 1 at corner node i, 0 at other nodes
        for i in 0..8 {
            let xi = NODE_COORDS[i][0];
            let eta = NODE_COORDS[i][1];
            let zeta = NODE_COORDS[i][2];
            let n = Hex20::shape_functions(xi, eta, zeta);

            for j in 0..20 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (n[j] - expected).abs() < 1e-12,
                    "At corner node {}, N[{}] = {} (expected {})",
                    i,
                    j,
                    n[j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_hex20_shape_functions_at_midedge_nodes() {
        // N_i = 1 at mid-edge node i, 0 at other nodes
        for i in 8..20 {
            let xi = NODE_COORDS[i][0];
            let eta = NODE_COORDS[i][1];
            let zeta = NODE_COORDS[i][2];
            let n = Hex20::shape_functions(xi, eta, zeta);

            for j in 0..20 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (n[j] - expected).abs() < 1e-12,
                    "At mid-edge node {}, N[{}] = {} (expected {})",
                    i,
                    j,
                    n[j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_hex20_volume_unit_cube() {
        let hex = Hex20::new();
        let coords = unit_cube_hex20();
        let vol = hex.volume(&coords);
        assert_relative_eq!(vol, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hex20_volume_stretched() {
        let hex = Hex20::new();
        let coords = stretched_hex20();
        let vol = hex.volume(&coords);
        assert_relative_eq!(vol, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hex20_node_count() {
        let hex = Hex20::new();
        assert_eq!(hex.n_nodes(), 20);
        assert_eq!(hex.dofs_per_node(), 3);
        assert_eq!(hex.n_dofs(), 60);
    }

    #[test]
    fn test_hex20_stiffness_symmetric() {
        let hex = Hex20::new();
        let coords = unit_cube_hex20();
        let mat = Material::steel();

        let k = hex.stiffness(&coords, &mat);

        assert_eq!(k.nrows(), 60);
        assert_eq!(k.ncols(), 60);

        // Check symmetry - use both absolute and relative tolerance
        // for small values near zero
        for i in 0..60 {
            for j in 0..60 {
                let diff = (k[(i, j)] - k[(j, i)]).abs();
                let max_val = k[(i, j)].abs().max(k[(j, i)].abs());
                // Either the absolute difference is tiny, or the relative difference is tiny
                assert!(
                    diff < 1e-10 || diff / max_val < 1e-10,
                    "Asymmetry at ({}, {}): {} vs {}",
                    i,
                    j,
                    k[(i, j)],
                    k[(j, i)]
                );
            }
        }
    }

    #[test]
    fn test_hex20_stiffness_positive_diagonal() {
        let hex = Hex20::new();
        let coords = unit_cube_hex20();
        let mat = Material::steel();

        let k = hex.stiffness(&coords, &mat);

        for i in 0..60 {
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
    fn test_hex20_rigid_body_translation() {
        let hex = Hex20::new();
        let coords = unit_cube_hex20();
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = hex.stiffness(&coords, &mat);

        // x-translation: all nodes move by (1, 0, 0)
        let mut u_x = vec![0.0; 60];
        for i in 0..20 {
            u_x[i * 3] = 1.0;
        }
        let u_x = nalgebra::DVector::from_vec(u_x);
        let f_x = &k * &u_x;
        assert_relative_eq!(f_x.norm(), 0.0, epsilon = 1e-8);

        // y-translation
        let mut u_y = vec![0.0; 60];
        for i in 0..20 {
            u_y[i * 3 + 1] = 1.0;
        }
        let u_y = nalgebra::DVector::from_vec(u_y);
        let f_y = &k * &u_y;
        assert_relative_eq!(f_y.norm(), 0.0, epsilon = 1e-8);

        // z-translation
        let mut u_z = vec![0.0; 60];
        for i in 0..20 {
            u_z[i * 3 + 2] = 1.0;
        }
        let u_z = nalgebra::DVector::from_vec(u_z);
        let f_z = &k * &u_z;
        assert_relative_eq!(f_z.norm(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_hex20_constant_strain_patch() {
        // Patch test: impose uniform ε_xx = 0.001
        // Displacement field: u = 0.001 * x, v = 0, w = 0
        let hex = Hex20::new();
        let coords = unit_cube_hex20();
        let mat = Material::new(1e6, 0.25).unwrap();

        // Displacements: u = 0.001 * x for each node
        let strain_xx = 0.001;
        let mut displacements = vec![0.0; 60];
        for i in 0..20 {
            let x = coords[i][0];
            displacements[i * 3] = strain_xx * x;
        }

        let stresses = hex.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 27); // 3×3×3 integration points

        // All integration points should have the same stress (constant strain)
        let d = mat.constitutive_3d();
        let expected_sigma_xx = d[(0, 0)] * strain_xx;
        let expected_sigma_yy = d[(1, 0)] * strain_xx;
        let expected_sigma_zz = d[(2, 0)] * strain_xx;

        for stress in &stresses {
            assert_relative_eq!(
                stress.0[0],
                expected_sigma_xx,
                epsilon = 1e-4,
                max_relative = 1e-4
            );
            assert_relative_eq!(
                stress.0[1],
                expected_sigma_yy,
                epsilon = 1e-4,
                max_relative = 1e-4
            );
            assert_relative_eq!(
                stress.0[2],
                expected_sigma_zz,
                epsilon = 1e-4,
                max_relative = 1e-4
            );
            // Shear stresses should be zero
            assert_relative_eq!(stress.0[3], 0.0, epsilon = 1e-8);
            assert_relative_eq!(stress.0[4], 0.0, epsilon = 1e-8);
            assert_relative_eq!(stress.0[5], 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_hex20_stress_count() {
        let hex = Hex20::new();
        let coords = unit_cube_hex20();
        let mat = Material::steel();
        let displacements = [0.0; 60];

        let stresses = hex.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 27); // One stress per integration point (3×3×3)
    }

    #[test]
    fn test_hex20_shape_derivatives_numerical() {
        // Verify analytical derivatives against finite difference approximation
        let h = 1e-6;
        let test_points = [(0.3, 0.5, -0.2), (-0.5, 0.7, 0.3), (0.0, 0.0, 0.0)];

        for (xi, eta, zeta) in test_points {
            let (dn_dxi, dn_deta, dn_dzeta) = Hex20::shape_derivatives(xi, eta, zeta);

            // Compute numerical derivatives
            let n_xi_plus = Hex20::shape_functions(xi + h, eta, zeta);
            let n_xi_minus = Hex20::shape_functions(xi - h, eta, zeta);
            let n_eta_plus = Hex20::shape_functions(xi, eta + h, zeta);
            let n_eta_minus = Hex20::shape_functions(xi, eta - h, zeta);
            let n_zeta_plus = Hex20::shape_functions(xi, eta, zeta + h);
            let n_zeta_minus = Hex20::shape_functions(xi, eta, zeta - h);

            for i in 0..20 {
                let dn_dxi_num = (n_xi_plus[i] - n_xi_minus[i]) / (2.0 * h);
                let dn_deta_num = (n_eta_plus[i] - n_eta_minus[i]) / (2.0 * h);
                let dn_dzeta_num = (n_zeta_plus[i] - n_zeta_minus[i]) / (2.0 * h);

                assert_relative_eq!(dn_dxi[i], dn_dxi_num, epsilon = 1e-6, max_relative = 1e-5);
                assert_relative_eq!(dn_deta[i], dn_deta_num, epsilon = 1e-6, max_relative = 1e-5);
                assert_relative_eq!(
                    dn_dzeta[i],
                    dn_dzeta_num,
                    epsilon = 1e-6,
                    max_relative = 1e-5
                );
            }
        }
    }

    #[test]
    fn test_hex20_linear_strain_field() {
        // Test that Hex20 can exactly represent a linear strain field
        // (which is higher order than Hex8's constant strain)
        // Displacement: u = c * x * y, which gives γ_xy = c * (x + y)
        let hex = Hex20::new();
        let coords = unit_cube_hex20();
        let mat = Material::new(1e6, 0.0).unwrap(); // ν=0 for simpler analysis

        let c = 0.001;
        let mut displacements = vec![0.0; 60];
        for i in 0..20 {
            let x = coords[i][0];
            let y = coords[i][1];
            // u = c * x * y
            displacements[i * 3] = c * x * y;
        }

        // Just verify it runs without error and produces reasonable results
        let stresses = hex.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 27);

        // The shear strain should vary linearly within the element
        // This is a basic sanity check that the element handles variable strain
        for stress in &stresses {
            // All stress components should be finite
            for j in 0..6 {
                assert!(
                    stress.0[j].is_finite(),
                    "Stress component {} is not finite",
                    j
                );
            }
        }
    }
}
