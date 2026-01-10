//! Core data types for FEA operations.
//!
//! This module defines fundamental types used throughout MOPS:
//! - Geometric primitives (points, vectors)
//! - Stress and strain tensors
//! - Degree of freedom specifications

use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};

/// A point in 3D space.
pub type Point3 = Vector3<f64>;

/// A 3D vector (displacement, force, etc.).
pub type Vec3 = Vector3<f64>;

/// Symmetric stress tensor in Voigt notation.
///
/// Components are ordered as: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressTensor(pub Vector6<f64>);

impl StressTensor {
    /// Create a new stress tensor from Voigt components.
    pub fn new(components: [f64; 6]) -> Self {
        Self(Vector6::from_row_slice(&components))
    }

    /// Zero stress state.
    pub fn zero() -> Self {
        Self(Vector6::zeros())
    }

    /// Compute von Mises equivalent stress.
    pub fn von_mises(&self) -> f64 {
        let s = &self.0;
        let s_xx = s[0];
        let s_yy = s[1];
        let s_zz = s[2];
        let t_xy = s[3];
        let t_yz = s[4];
        let t_xz = s[5];

        let term1 = (s_xx - s_yy).powi(2) + (s_yy - s_zz).powi(2) + (s_zz - s_xx).powi(2);
        let term2 = 6.0 * (t_xy.powi(2) + t_yz.powi(2) + t_xz.powi(2));

        ((term1 + term2) / 2.0).sqrt()
    }

    /// Compute hydrostatic (mean) stress.
    pub fn hydrostatic(&self) -> f64 {
        (self.0[0] + self.0[1] + self.0[2]) / 3.0
    }

    /// Extract the full 3x3 symmetric stress matrix.
    pub fn to_matrix(&self) -> Matrix3<f64> {
        let s = &self.0;
        Matrix3::new(
            s[0], s[3], s[5],
            s[3], s[1], s[4],
            s[5], s[4], s[2],
        )
    }
}

/// Symmetric strain tensor in Voigt notation.
///
/// Components are ordered as: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz]
/// where γ = 2ε for engineering shear strain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrainTensor(pub Vector6<f64>);

impl StrainTensor {
    /// Create a new strain tensor from Voigt components.
    pub fn new(components: [f64; 6]) -> Self {
        Self(Vector6::from_row_slice(&components))
    }

    /// Zero strain state.
    pub fn zero() -> Self {
        Self(Vector6::zeros())
    }

    /// Compute volumetric strain.
    pub fn volumetric(&self) -> f64 {
        self.0[0] + self.0[1] + self.0[2]
    }

    /// Extract the full 3x3 symmetric strain matrix.
    pub fn to_matrix(&self) -> Matrix3<f64> {
        let e = &self.0;
        // Note: off-diagonal terms are γ/2 = ε
        Matrix3::new(
            e[0],       e[3] / 2.0, e[5] / 2.0,
            e[3] / 2.0, e[1],       e[4] / 2.0,
            e[5] / 2.0, e[4] / 2.0, e[2],
        )
    }
}

/// Constitutive matrix (material stiffness) in Voigt notation.
///
/// Maps strain tensor to stress tensor: σ = D * ε
pub type ConstitutiveMatrix = Matrix6<f64>;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_von_mises_uniaxial() {
        // Pure uniaxial tension: σ_xx = 100 MPa
        let stress = StressTensor::new([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(stress.von_mises(), 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_von_mises_pure_shear() {
        // Pure shear: τ_xy = 100 MPa
        // von Mises = √3 * τ ≈ 173.2 MPa
        let stress = StressTensor::new([0.0, 0.0, 0.0, 100.0, 0.0, 0.0]);
        assert_relative_eq!(stress.von_mises(), 100.0 * 3.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_hydrostatic_stress() {
        let stress = StressTensor::new([100.0, 200.0, 300.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(stress.hydrostatic(), 200.0, epsilon = 1e-10);
    }

    #[test]
    fn test_volumetric_strain() {
        let strain = StrainTensor::new([0.001, 0.002, 0.003, 0.0, 0.0, 0.0]);
        assert_relative_eq!(strain.volumetric(), 0.006, epsilon = 1e-15);
    }
}
