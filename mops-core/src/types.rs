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
        Matrix3::new(s[0], s[3], s[5], s[3], s[1], s[4], s[5], s[4], s[2])
    }

    /// Compute principal stresses by eigenvalue decomposition.
    ///
    /// Returns principal stresses in descending order: (σ₁, σ₂, σ₃) where σ₁ ≥ σ₂ ≥ σ₃.
    /// These are the eigenvalues of the stress tensor matrix.
    pub fn principal_stresses(&self) -> (f64, f64, f64) {
        let matrix = self.to_matrix();
        let eigen = matrix.symmetric_eigen();
        let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        (eigenvalues[0], eigenvalues[1], eigenvalues[2])
    }

    /// Compute principal directions (eigenvectors of stress tensor).
    ///
    /// Returns 3 unit vectors corresponding to the principal stress directions.
    /// The vectors are ordered to correspond with principal stresses (σ₁, σ₂, σ₃).
    pub fn principal_directions(&self) -> [Vec3; 3] {
        let matrix = self.to_matrix();
        let eigen = matrix.symmetric_eigen();

        // Get eigenvalues and pair with eigenvectors
        let mut pairs: Vec<(f64, Vec3)> = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, eigen.eigenvectors.column(i).into()))
            .collect();

        // Sort by eigenvalue descending
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        [pairs[0].1, pairs[1].1, pairs[2].1]
    }

    /// Compute maximum shear stress.
    ///
    /// τ_max = (σ₁ - σ₃) / 2
    pub fn max_shear(&self) -> f64 {
        let (s1, _, s3) = self.principal_stresses();
        (s1 - s3) / 2.0
    }

    /// Compute Tresca (maximum shear stress) equivalent stress.
    ///
    /// σ_tresca = σ₁ - σ₃
    pub fn tresca(&self) -> f64 {
        let (s1, _, s3) = self.principal_stresses();
        s1 - s3
    }

    /// Compute deviatoric stress tensor.
    ///
    /// s = σ - σ_m * I where σ_m is the hydrostatic stress.
    pub fn deviatoric(&self) -> StressTensor {
        let p = self.hydrostatic();
        StressTensor::new([
            self.0[0] - p,
            self.0[1] - p,
            self.0[2] - p,
            self.0[3],
            self.0[4],
            self.0[5],
        ])
    }

    /// Compute pressure (negative hydrostatic stress).
    pub fn pressure(&self) -> f64 {
        -self.hydrostatic()
    }

    /// Compute stress intensity (maximum of principal stress differences).
    ///
    /// σ_intensity = max(|σ₁ - σ₂|, |σ₂ - σ₃|, |σ₃ - σ₁|)
    pub fn intensity(&self) -> f64 {
        let (s1, s2, s3) = self.principal_stresses();
        f64::max(
            (s1 - s2).abs(),
            f64::max((s2 - s3).abs(), (s3 - s1).abs()),
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
            e[0],
            e[3] / 2.0,
            e[5] / 2.0,
            e[3] / 2.0,
            e[1],
            e[4] / 2.0,
            e[5] / 2.0,
            e[4] / 2.0,
            e[2],
        )
    }

    /// Compute von Mises equivalent strain.
    ///
    /// This is the strain analog of von Mises stress, useful for plasticity analysis.
    /// ε_eq = √(2/3) * √(e_ij * e_ij) where e is the deviatoric strain tensor.
    pub fn von_mises(&self) -> f64 {
        let e = &self.0;
        let e_xx = e[0];
        let e_yy = e[1];
        let e_zz = e[2];
        // Engineering shear strains (γ = 2ε for off-diagonal)
        let g_xy = e[3];
        let g_yz = e[4];
        let g_xz = e[5];

        // For equivalent strain: ε_eq = sqrt(2/3) * sqrt(ε':ε')
        // where ε' is deviatoric strain
        // Using direct formula for equivalent strain:
        // ε_eq = (2/3) * sqrt[ (ε_xx - ε_yy)² + (ε_yy - ε_zz)² + (ε_zz - ε_xx)²
        //                      + 6*(ε_xy² + ε_yz² + ε_xz²) ] / sqrt(2)
        // With engineering shear strains (γ = 2ε):
        let term1 = (e_xx - e_yy).powi(2) + (e_yy - e_zz).powi(2) + (e_zz - e_xx).powi(2);
        let term2 = 1.5 * (g_xy.powi(2) + g_yz.powi(2) + g_xz.powi(2));

        ((term1 + term2) / 2.0).sqrt() * (2.0 / 3.0_f64).sqrt()
    }

    /// Compute principal strains by eigenvalue decomposition.
    ///
    /// Returns principal strains in descending order: (ε₁, ε₂, ε₃) where ε₁ ≥ ε₂ ≥ ε₃.
    pub fn principal_strains(&self) -> (f64, f64, f64) {
        let matrix = self.to_matrix();
        let eigen = matrix.symmetric_eigen();
        let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        (eigenvalues[0], eigenvalues[1], eigenvalues[2])
    }

    /// Compute maximum shear strain.
    ///
    /// γ_max = ε₁ - ε₃
    pub fn max_shear(&self) -> f64 {
        let (e1, _, e3) = self.principal_strains();
        e1 - e3
    }

    /// Compute deviatoric strain tensor.
    ///
    /// e' = ε - ε_m * I where ε_m = ε_vol / 3
    pub fn deviatoric(&self) -> StrainTensor {
        let em = self.volumetric() / 3.0;
        StrainTensor::new([
            self.0[0] - em,
            self.0[1] - em,
            self.0[2] - em,
            self.0[3],
            self.0[4],
            self.0[5],
        ])
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

    #[test]
    fn test_principal_stresses_uniaxial() {
        // Pure uniaxial tension
        let stress = StressTensor::new([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (s1, s2, s3) = stress.principal_stresses();
        assert_relative_eq!(s1, 100.0, epsilon = 1e-10);
        assert_relative_eq!(s2, 0.0, epsilon = 1e-10);
        assert_relative_eq!(s3, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_principal_stresses_biaxial() {
        // Biaxial stress state
        let stress = StressTensor::new([100.0, 50.0, 0.0, 0.0, 0.0, 0.0]);
        let (s1, s2, s3) = stress.principal_stresses();
        assert_relative_eq!(s1, 100.0, epsilon = 1e-10);
        assert_relative_eq!(s2, 50.0, epsilon = 1e-10);
        assert_relative_eq!(s3, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_principal_stresses_with_shear() {
        // 2D stress state with shear: σ_xx = 100, σ_yy = 0, τ_xy = 50
        // Principal stresses from Mohr's circle: s1 = 50 + sqrt(50² + 50²) ≈ 120.7
        // s2 = 50 - sqrt(50² + 50²) ≈ -20.7
        let stress = StressTensor::new([100.0, 0.0, 0.0, 50.0, 0.0, 0.0]);
        let (s1, s2, s3) = stress.principal_stresses();
        let center = 50.0;
        let radius = (50.0_f64.powi(2) + 50.0_f64.powi(2)).sqrt();
        assert_relative_eq!(s1, center + radius, epsilon = 1e-10);
        assert_relative_eq!(s2, 0.0, epsilon = 1e-10);
        assert_relative_eq!(s3, center - radius, epsilon = 1e-10);
    }

    #[test]
    fn test_tresca_equals_von_mises_uniaxial() {
        // For uniaxial tension, Tresca = von Mises
        let stress = StressTensor::new([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(stress.tresca(), stress.von_mises(), epsilon = 1e-10);
    }

    #[test]
    fn test_max_shear_uniaxial() {
        // τ_max = σ/2 for uniaxial tension
        let stress = StressTensor::new([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(stress.max_shear(), 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_deviatoric_stress() {
        let stress = StressTensor::new([100.0, 200.0, 300.0, 10.0, 20.0, 30.0]);
        let dev = stress.deviatoric();
        // Hydrostatic = 200, so deviatoric normal components are -100, 0, 100
        assert_relative_eq!(dev.0[0], -100.0, epsilon = 1e-10);
        assert_relative_eq!(dev.0[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dev.0[2], 100.0, epsilon = 1e-10);
        // Shear components unchanged
        assert_relative_eq!(dev.0[3], 10.0, epsilon = 1e-10);
        assert_relative_eq!(dev.0[4], 20.0, epsilon = 1e-10);
        assert_relative_eq!(dev.0[5], 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stress_intensity() {
        let stress = StressTensor::new([100.0, 50.0, 0.0, 0.0, 0.0, 0.0]);
        // Principal: 100, 50, 0
        // Intensity = max(|100-50|, |50-0|, |0-100|) = 100
        assert_relative_eq!(stress.intensity(), 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_principal_strains() {
        // Pure axial strain
        let strain = StrainTensor::new([0.001, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (e1, e2, e3) = strain.principal_strains();
        assert_relative_eq!(e1, 0.001, epsilon = 1e-15);
        assert_relative_eq!(e2, 0.0, epsilon = 1e-15);
        assert_relative_eq!(e3, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_equivalent_strain_uniaxial() {
        // For uniaxial strain ε_xx with ε_yy = ε_zz = -ν*ε_xx,
        // the equivalent strain should equal |ε_xx| (for ν=0.5 incompressible)
        // For pure axial strain (no Poisson effect), we use simplified formula
        let strain = StrainTensor::new([0.001, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let eq = strain.von_mises();
        // eq = sqrt(2/3) * sqrt[(e_xx - e_yy)² + ...] / sqrt(2)
        // = sqrt(2/3) * sqrt(2*e_xx²) / sqrt(2) = sqrt(2/3) * e_xx
        assert_relative_eq!(eq, 0.001 * (2.0 / 3.0_f64).sqrt(), epsilon = 1e-15);
    }

    #[test]
    fn test_max_shear_strain() {
        let strain = StrainTensor::new([0.001, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Max shear = e1 - e3 = 0.001 - 0 = 0.001
        assert_relative_eq!(strain.max_shear(), 0.001, epsilon = 1e-15);
    }

    #[test]
    fn test_deviatoric_strain() {
        let strain = StrainTensor::new([0.003, 0.002, 0.001, 0.0001, 0.0002, 0.0003]);
        let dev = strain.deviatoric();
        // volumetric = 0.006, mean = 0.002
        assert_relative_eq!(dev.0[0], 0.001, epsilon = 1e-15);
        assert_relative_eq!(dev.0[1], 0.0, epsilon = 1e-15);
        assert_relative_eq!(dev.0[2], -0.001, epsilon = 1e-15);
        // Engineering shear unchanged
        assert_relative_eq!(dev.0[3], 0.0001, epsilon = 1e-15);
        assert_relative_eq!(dev.0[4], 0.0002, epsilon = 1e-15);
        assert_relative_eq!(dev.0[5], 0.0003, epsilon = 1e-15);
    }
}
