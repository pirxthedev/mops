//! 4-node tetrahedron (Tet4) element.
//!
//! The Tet4 is the simplest 3D solid element with:
//! - 4 nodes at vertices
//! - 3 DOFs per node (u, v, w displacements)
//! - 12 total DOFs
//! - Constant strain/stress within element
//! - Single integration point at centroid
//!
//! # Shape Functions
//!
//! Linear shape functions in terms of barycentric coordinates (L1, L2, L3, L4):
//! - N_i = L_i where sum(L_i) = 1
//!
//! # Limitations
//!
//! - Volumetric locking in nearly incompressible materials (ν → 0.5)
//! - Low accuracy - requires fine meshes
//! - Shear locking in bending-dominated problems

use crate::element::Element;
use crate::material::Material;
use crate::types::{Point3, StressTensor};
use nalgebra::{DMatrix, Matrix3, Vector3, Vector6};

/// 4-node tetrahedral element (constant strain tetrahedron).
#[derive(Debug, Clone, Copy, Default)]
pub struct Tet4;

impl Tet4 {
    /// Create a new Tet4 element.
    pub fn new() -> Self {
        Self
    }

    /// Compute the B-matrix (strain-displacement) for constant strain.
    ///
    /// Returns the 6x12 B-matrix where ε = B * u
    /// and shape function derivatives b, c, d scaled by 6V.
    ///
    /// Also returns the element volume.
    fn compute_b_matrix(coords: &[Point3]) -> (DMatrix<f64>, f64) {
        assert_eq!(coords.len(), 4, "Tet4 requires exactly 4 nodal coordinates");

        // Node coordinates
        let x1 = coords[0][0];
        let y1 = coords[0][1];
        let z1 = coords[0][2];
        let x2 = coords[1][0];
        let y2 = coords[1][1];
        let z2 = coords[1][2];
        let x3 = coords[2][0];
        let y3 = coords[2][1];
        let z3 = coords[2][2];
        let x4 = coords[3][0];
        let y4 = coords[3][1];
        let z4 = coords[3][2];

        // Volume via triple product: V = det([x2-x1, x3-x1, x4-x1]) / 6
        let v21 = Vector3::new(x2 - x1, y2 - y1, z2 - z1);
        let v31 = Vector3::new(x3 - x1, y3 - y1, z3 - z1);
        let v41 = Vector3::new(x4 - x1, y4 - y1, z4 - z1);

        let volume_6_signed = v21.dot(&v31.cross(&v41));
        let volume = volume_6_signed.abs() / 6.0;

        // Shape function derivatives (b_i, c_i, d_i) where:
        // N_i = (a_i + b_i*x + c_i*y + d_i*z) / (6V)
        //
        // Using the standard formulation from FEA textbooks.
        // The derivatives are computed from the inverse of the Jacobian.
        //
        // For a tetrahedron with vertices (x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4):
        // We use the formula based on cofactors of the coordinate matrix.

        // Compute shape function derivatives: dN_i/dx, dN_i/dy, dN_i/dz
        // These are constants for the linear tetrahedron.
        //
        // The gradient of shape function N_i is:
        // ∇N_i = (1 / (6V)) * [b_i, c_i, d_i]
        //
        // where b_i, c_i, d_i are cofactors.

        // For simplicity, use the inverse Jacobian approach:
        // J = [[x2-x1, x3-x1, x4-x1],
        //      [y2-y1, y3-y1, y4-y1],
        //      [z2-z1, z3-z1, z4-z1]]
        //
        // dN/dξ = [[-1, 1, 0, 0],
        //          [-1, 0, 1, 0],
        //          [-1, 0, 0, 1]]
        //
        // dN/dx = J^(-T) * dN/dξ

        let j = Matrix3::new(
            x2 - x1,
            x3 - x1,
            x4 - x1,
            y2 - y1,
            y3 - y1,
            y4 - y1,
            z2 - z1,
            z3 - z1,
            z4 - z1,
        );

        let j_inv = j
            .try_inverse()
            .expect("Degenerate tetrahedron: Jacobian is singular");
        let j_inv_t = j_inv.transpose();

        // dN/dξ for nodes 1-4 in natural coords (L1, L2, L3, L4 = 1 - L1 - L2 - L3)
        // Actually for Tet4, parametric coords are (ξ, η, ζ) where:
        // Node 1: (0, 0, 0) -> N1 = 1 - ξ - η - ζ
        // Node 2: (1, 0, 0) -> N2 = ξ
        // Node 3: (0, 1, 0) -> N3 = η
        // Node 4: (0, 0, 1) -> N4 = ζ
        //
        // dN1/dξ = -1, dN1/dη = -1, dN1/dζ = -1
        // dN2/dξ =  1, dN2/dη =  0, dN2/dζ =  0
        // dN3/dξ =  0, dN3/dη =  1, dN3/dζ =  0
        // dN4/dξ =  0, dN4/dη =  0, dN4/dζ =  1

        // dN/dx = J^(-T) * dN/dξ
        // For each node, dN_i/dx is the dot product of J^(-T) row 1 with dN_i/dξ column
        // But actually we have: [dN/dx, dN/dy, dN/dz]^T = J^(-T) * [dN/dξ, dN/dη, dN/dζ]^T

        let dn1_dxi = Vector3::new(-1.0, -1.0, -1.0);
        let dn2_dxi = Vector3::new(1.0, 0.0, 0.0);
        let dn3_dxi = Vector3::new(0.0, 1.0, 0.0);
        let dn4_dxi = Vector3::new(0.0, 0.0, 1.0);

        let dn1_dx = &j_inv_t * dn1_dxi;
        let dn2_dx = &j_inv_t * dn2_dxi;
        let dn3_dx = &j_inv_t * dn3_dxi;
        let dn4_dx = &j_inv_t * dn4_dxi;

        // Shape function derivatives
        let (b1, c1, d1) = (dn1_dx[0], dn1_dx[1], dn1_dx[2]);
        let (b2, c2, d2) = (dn2_dx[0], dn2_dx[1], dn2_dx[2]);
        let (b3, c3, d3) = (dn3_dx[0], dn3_dx[1], dn3_dx[2]);
        let (b4, c4, d4) = (dn4_dx[0], dn4_dx[1], dn4_dx[2]);

        // B-matrix construction (6x12)
        // ε = [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz]^T = B * u
        //
        // Row structure for each node column block [u_i, v_i, w_i]:
        // ε_xx: ∂u/∂x         -> [b_i, 0, 0]
        // ε_yy: ∂v/∂y         -> [0, c_i, 0]
        // ε_zz: ∂w/∂z         -> [0, 0, d_i]
        // γ_xy: ∂u/∂y + ∂v/∂x -> [c_i, b_i, 0]
        // γ_yz: ∂v/∂z + ∂w/∂y -> [0, d_i, c_i]
        // γ_xz: ∂u/∂z + ∂w/∂x -> [d_i, 0, b_i]
        //
        // Note: b, c, d are already the actual derivatives (dN/dx, dN/dy, dN/dz)

        let mut b = DMatrix::zeros(6, 12);

        // Node 1 (columns 0, 1, 2)
        b[(0, 0)] = b1;
        b[(1, 1)] = c1;
        b[(2, 2)] = d1;
        b[(3, 0)] = c1;
        b[(3, 1)] = b1;
        b[(4, 1)] = d1;
        b[(4, 2)] = c1;
        b[(5, 0)] = d1;
        b[(5, 2)] = b1;

        // Node 2 (columns 3, 4, 5)
        b[(0, 3)] = b2;
        b[(1, 4)] = c2;
        b[(2, 5)] = d2;
        b[(3, 3)] = c2;
        b[(3, 4)] = b2;
        b[(4, 4)] = d2;
        b[(4, 5)] = c2;
        b[(5, 3)] = d2;
        b[(5, 5)] = b2;

        // Node 3 (columns 6, 7, 8)
        b[(0, 6)] = b3;
        b[(1, 7)] = c3;
        b[(2, 8)] = d3;
        b[(3, 6)] = c3;
        b[(3, 7)] = b3;
        b[(4, 7)] = d3;
        b[(4, 8)] = c3;
        b[(5, 6)] = d3;
        b[(5, 8)] = b3;

        // Node 4 (columns 9, 10, 11)
        b[(0, 9)] = b4;
        b[(1, 10)] = c4;
        b[(2, 11)] = d4;
        b[(3, 9)] = c4;
        b[(3, 10)] = b4;
        b[(4, 10)] = d4;
        b[(4, 11)] = c4;
        b[(5, 9)] = d4;
        b[(5, 11)] = b4;

        (b, volume)
    }
}

impl Element for Tet4 {
    fn n_nodes(&self) -> usize {
        4
    }

    fn dofs_per_node(&self) -> usize {
        3
    }

    fn stiffness(&self, coords: &[Point3], material: &Material) -> DMatrix<f64> {
        let (b, volume) = Self::compute_b_matrix(coords);

        // D is the 6x6 constitutive matrix
        let d = material.constitutive_3d();

        // K = V * B^T * D * B (12x12)
        // Using: K = B^T * (D * B) * V
        let db = &d * &b; // 6x12
        let k = b.transpose() * db * volume; // 12x12

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
            12,
            "Tet4 requires 12 displacement DOFs"
        );

        let (b, _volume) = Self::compute_b_matrix(coords);
        let d = material.constitutive_3d();

        // ε = B * u
        let u = nalgebra::DVector::from_row_slice(displacements);
        let strain = &b * &u; // 6x1

        // σ = D * ε
        let stress_vec = &d * Vector6::from_iterator(strain.iter().cloned());

        // Single stress tensor (constant strain element)
        vec![StressTensor(stress_vec)]
    }

    fn volume(&self, coords: &[Point3]) -> f64 {
        assert_eq!(coords.len(), 4, "Tet4 requires exactly 4 nodal coordinates");

        let v21 = coords[1] - coords[0];
        let v31 = coords[2] - coords[0];
        let v41 = coords[3] - coords[0];

        v21.dot(&v31.cross(&v41)).abs() / 6.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn unit_tetrahedron() -> Vec<Point3> {
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]
    }

    #[test]
    fn test_tet4_volume_unit() {
        let tet = Tet4::new();
        let coords = unit_tetrahedron();
        let vol = tet.volume(&coords);
        // Unit tetrahedron volume = 1/6
        assert_relative_eq!(vol, 1.0 / 6.0, epsilon = 1e-14);
    }

    #[test]
    fn test_tet4_volume_scaled() {
        let tet = Tet4::new();
        // Scale by 2 in each direction -> volume * 8
        let coords = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(0.0, 0.0, 2.0),
        ];
        let vol = tet.volume(&coords);
        assert_relative_eq!(vol, 8.0 / 6.0, epsilon = 1e-14);
    }

    #[test]
    fn test_tet4_stiffness_symmetric() {
        let tet = Tet4::new();
        let coords = unit_tetrahedron();
        let mat = Material::steel();

        let k = tet.stiffness(&coords, &mat);

        // Stiffness matrix must be symmetric
        assert_eq!(k.nrows(), 12);
        assert_eq!(k.ncols(), 12);
        for i in 0..12 {
            for j in 0..12 {
                assert_relative_eq!(k[(i, j)], k[(j, i)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_tet4_stiffness_positive_diagonal() {
        let tet = Tet4::new();
        let coords = unit_tetrahedron();
        let mat = Material::steel();

        let k = tet.stiffness(&coords, &mat);

        // Diagonal elements should be positive
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
    fn test_tet4_rigid_body_modes() {
        let tet = Tet4::new();
        let coords = unit_tetrahedron();
        // Use unit material to avoid large numerical values
        let mat = Material::new(1.0, 0.3).unwrap();

        let k = tet.stiffness(&coords, &mat);

        // Pure translation should produce zero strain (and thus zero force)
        // u = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0] (x-translation)
        let u_x = nalgebra::DVector::from_vec(vec![
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ]);
        let f_x = &k * &u_x;
        let f_norm = f_x.norm();
        assert_relative_eq!(f_norm, 0.0, epsilon = 1e-12);

        // y-translation
        let u_y = nalgebra::DVector::from_vec(vec![
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ]);
        let f_y = &k * &u_y;
        assert_relative_eq!(f_y.norm(), 0.0, epsilon = 1e-12);

        // z-translation
        let u_z = nalgebra::DVector::from_vec(vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ]);
        let f_z = &k * &u_z;
        assert_relative_eq!(f_z.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_tet4_constant_strain_patch() {
        // Patch test: impose a constant strain field and verify correct stress
        let tet = Tet4::new();
        let coords = unit_tetrahedron();
        let mat = Material::new(1e6, 0.25).unwrap(); // Simple material

        // Impose uniform ε_xx = 0.001 (all other strains = 0)
        // For ε_xx = du/dx = 0.001, displacements are u = 0.001 * x
        // Node displacements [u, v, w]:
        // Node 0 (0,0,0): [0, 0, 0]
        // Node 1 (1,0,0): [0.001, 0, 0]
        // Node 2 (0,1,0): [0, 0, 0]
        // Node 3 (0,0,1): [0, 0, 0]
        let displacements = [
            0.0, 0.0, 0.0, // node 0
            0.001, 0.0, 0.0, // node 1
            0.0, 0.0, 0.0, // node 2
            0.0, 0.0, 0.0, // node 3
        ];

        let stresses = tet.stress(&coords, &displacements, &mat);
        assert_eq!(stresses.len(), 1); // Constant strain element

        let stress = &stresses[0];

        // Expected stress from σ = D * ε
        // With E=1e6, ν=0.25:
        // D[0,0] = E(1-ν)/((1+ν)(1-2ν)) = 1e6 * 0.75 / (1.25 * 0.5) = 1.2e6
        // D[0,1] = Eν/((1+ν)(1-2ν)) = 1e6 * 0.25 / (1.25 * 0.5) = 0.4e6
        // σ_xx = D[0,0] * ε_xx = 1.2e6 * 0.001 = 1200
        // σ_yy = σ_zz = D[0,1] * ε_xx = 0.4e6 * 0.001 = 400

        let d = mat.constitutive_3d();
        let expected_sigma_xx = d[(0, 0)] * 0.001;
        let expected_sigma_yy = d[(1, 0)] * 0.001;
        let expected_sigma_zz = d[(2, 0)] * 0.001;

        assert_relative_eq!(stress.0[0], expected_sigma_xx, epsilon = 1e-3);
        assert_relative_eq!(stress.0[1], expected_sigma_yy, epsilon = 1e-3);
        assert_relative_eq!(stress.0[2], expected_sigma_zz, epsilon = 1e-3);
        assert_relative_eq!(stress.0[3], 0.0, epsilon = 1e-3); // τ_xy
        assert_relative_eq!(stress.0[4], 0.0, epsilon = 1e-3); // τ_yz
        assert_relative_eq!(stress.0[5], 0.0, epsilon = 1e-3); // τ_xz
    }

    #[test]
    fn test_tet4_node_count() {
        let tet = Tet4::new();
        assert_eq!(tet.n_nodes(), 4);
        assert_eq!(tet.dofs_per_node(), 3);
        assert_eq!(tet.n_dofs(), 12);
    }
}
