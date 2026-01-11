//! Material property definitions.
//!
//! Supports isotropic linear elastic materials for the initial implementation.
//! Future extensions will include orthotropic and anisotropic materials.

use crate::types::ConstitutiveMatrix;
use crate::error::{Error, Result};
use nalgebra::Matrix6;

/// Material properties for structural analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    /// Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Poisson's ratio (dimensionless).
    pub poissons_ratio: f64,
    /// Mass density (kg/m³), optional for static analysis.
    pub density: Option<f64>,
}

impl Material {
    /// Create a new isotropic linear elastic material.
    ///
    /// # Arguments
    ///
    /// * `youngs_modulus` - Young's modulus E (Pa)
    /// * `poissons_ratio` - Poisson's ratio ν (dimensionless, -1 < ν < 0.5)
    ///
    /// # Errors
    ///
    /// Returns error if material properties are physically invalid.
    pub fn new(youngs_modulus: f64, poissons_ratio: f64) -> Result<Self> {
        if youngs_modulus <= 0.0 {
            return Err(Error::InvalidMaterial(
                "Young's modulus must be positive".into(),
            ));
        }
        if poissons_ratio <= -1.0 || poissons_ratio >= 0.5 {
            return Err(Error::InvalidMaterial(
                "Poisson's ratio must be in range (-1, 0.5)".into(),
            ));
        }
        Ok(Self {
            youngs_modulus,
            poissons_ratio,
            density: None,
        })
    }

    /// Create a material with density specified.
    pub fn with_density(mut self, density: f64) -> Result<Self> {
        if density <= 0.0 {
            return Err(Error::InvalidMaterial("Density must be positive".into()));
        }
        self.density = Some(density);
        Ok(self)
    }

    /// Shear modulus G = E / (2(1 + ν)).
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))
    }

    /// Bulk modulus K = E / (3(1 - 2ν)).
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))
    }

    /// Lamé's first parameter λ = Eν / ((1+ν)(1-2ν)).
    pub fn lame_lambda(&self) -> f64 {
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;
        e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    }

    /// Lamé's second parameter μ = G (shear modulus).
    pub fn lame_mu(&self) -> f64 {
        self.shear_modulus()
    }

    /// 3D constitutive matrix for isotropic linear elasticity.
    ///
    /// Returns the 6x6 matrix D such that σ = D * ε in Voigt notation.
    pub fn constitutive_3d(&self) -> ConstitutiveMatrix {
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;

        let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let c11 = factor * (1.0 - nu);
        let c12 = factor * nu;
        let c44 = factor * (1.0 - 2.0 * nu) / 2.0; // = G

        Matrix6::new(
            c11, c12, c12, 0.0, 0.0, 0.0,
            c12, c11, c12, 0.0, 0.0, 0.0,
            c12, c12, c11, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, c44, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, c44, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, c44,
        )
    }

    /// Plane stress constitutive matrix (for 2D elements).
    ///
    /// Returns a 3x3 matrix for [σ_xx, σ_yy, τ_xy] = D * [ε_xx, ε_yy, γ_xy].
    pub fn constitutive_plane_stress(&self) -> nalgebra::Matrix3<f64> {
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;

        let factor = e / (1.0 - nu * nu);

        nalgebra::Matrix3::new(
            factor,         factor * nu, 0.0,
            factor * nu,    factor,      0.0,
            0.0,            0.0,         factor * (1.0 - nu) / 2.0,
        )
    }

    /// Plane strain constitutive matrix (for 2D elements).
    ///
    /// Returns a 3x3 matrix for [σ_xx, σ_yy, τ_xy] = D * [ε_xx, ε_yy, γ_xy].
    pub fn constitutive_plane_strain(&self) -> nalgebra::Matrix3<f64> {
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;

        let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let c11 = factor * (1.0 - nu);
        let c12 = factor * nu;
        let c44 = factor * (1.0 - 2.0 * nu) / 2.0;

        nalgebra::Matrix3::new(
            c11, c12, 0.0,
            c12, c11, 0.0,
            0.0, 0.0, c44,
        )
    }

    /// Axisymmetric constitutive matrix (for bodies of revolution).
    ///
    /// For axisymmetric problems, the strain vector has 4 components:
    /// ε = [ε_rr, ε_zz, ε_θθ, γ_rz]^T
    ///
    /// Returns a 4x4 matrix D such that:
    /// [σ_rr, σ_zz, σ_θθ, τ_rz]^T = D * [ε_rr, ε_zz, ε_θθ, γ_rz]^T
    ///
    /// The hoop strain ε_θθ = u_r / r is coupled to radial and axial strains
    /// through Poisson's ratio.
    pub fn constitutive_axisymmetric(&self) -> nalgebra::Matrix4<f64> {
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;

        let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let c11 = factor * (1.0 - nu);
        let c12 = factor * nu;
        let c44 = factor * (1.0 - 2.0 * nu) / 2.0; // = G (shear modulus)

        // D matrix for axisymmetric:
        // [σ_rr ]   [c11 c12 c12  0 ] [ε_rr ]
        // [σ_zz ] = [c12 c11 c12  0 ] [ε_zz ]
        // [σ_θθ ]   [c12 c12 c11  0 ] [ε_θθ ]
        // [τ_rz ]   [ 0   0   0  c44] [γ_rz ]
        nalgebra::Matrix4::new(
            c11, c12, c12, 0.0,
            c12, c11, c12, 0.0,
            c12, c12, c11, 0.0,
            0.0, 0.0, 0.0, c44,
        )
    }
}

/// Common material presets.
impl Material {
    /// Structural steel (E = 200 GPa, ν = 0.3, ρ = 7850 kg/m³).
    pub fn steel() -> Self {
        Self {
            youngs_modulus: 200e9,
            poissons_ratio: 0.3,
            density: Some(7850.0),
        }
    }

    /// Aluminum 6061-T6 (E = 68.9 GPa, ν = 0.33, ρ = 2700 kg/m³).
    pub fn aluminum() -> Self {
        Self {
            youngs_modulus: 68.9e9,
            poissons_ratio: 0.33,
            density: Some(2700.0),
        }
    }

    /// Titanium Ti-6Al-4V (E = 113.8 GPa, ν = 0.342, ρ = 4430 kg/m³).
    pub fn titanium() -> Self {
        Self {
            youngs_modulus: 113.8e9,
            poissons_ratio: 0.342,
            density: Some(4430.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_material_creation() {
        let mat = Material::new(200e9, 0.3).unwrap();
        assert_relative_eq!(mat.youngs_modulus, 200e9);
        assert_relative_eq!(mat.poissons_ratio, 0.3);
    }

    #[test]
    fn test_invalid_youngs_modulus() {
        assert!(Material::new(-100e9, 0.3).is_err());
        assert!(Material::new(0.0, 0.3).is_err());
    }

    #[test]
    fn test_invalid_poissons_ratio() {
        assert!(Material::new(200e9, 0.5).is_err());
        assert!(Material::new(200e9, -1.0).is_err());
        assert!(Material::new(200e9, 0.6).is_err());
    }

    #[test]
    fn test_shear_modulus() {
        let mat = Material::steel();
        // G = E / (2(1+ν)) = 200e9 / (2 * 1.3) ≈ 76.92 GPa
        let expected_g = 200e9 / (2.0 * 1.3);
        assert_relative_eq!(mat.shear_modulus(), expected_g, epsilon = 1e-6);
    }

    #[test]
    fn test_bulk_modulus() {
        let mat = Material::steel();
        // K = E / (3(1-2ν)) = 200e9 / (3 * 0.4) ≈ 166.67 GPa
        let expected_k = 200e9 / (3.0 * 0.4);
        assert_relative_eq!(mat.bulk_modulus(), expected_k, epsilon = 1e-6);
    }

    #[test]
    fn test_constitutive_symmetry() {
        let mat = Material::steel();
        let d = mat.constitutive_3d();
        // Constitutive matrix should be symmetric
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(d[(i, j)], d[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_axisymmetric_constitutive_symmetry() {
        let mat = Material::steel();
        let d = mat.constitutive_axisymmetric();
        // Constitutive matrix should be symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(d[(i, j)], d[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_axisymmetric_constitutive_positive_definite() {
        let mat = Material::steel();
        let d = mat.constitutive_axisymmetric();
        // Check diagonal elements are positive
        for i in 0..4 {
            assert!(d[(i, i)] > 0.0, "D[{},{}] = {} should be positive", i, i, d[(i, i)]);
        }
    }

    #[test]
    fn test_axisymmetric_constitutive_values() {
        // Verify specific values for steel
        let mat = Material::steel();
        let d = mat.constitutive_axisymmetric();

        let e = 200e9;
        let nu = 0.3;
        let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let c11 = factor * (1.0 - nu);
        let c12 = factor * nu;
        let c44 = factor * (1.0 - 2.0 * nu) / 2.0;

        // Check diagonal terms
        assert_relative_eq!(d[(0, 0)], c11, epsilon = 1e-3);
        assert_relative_eq!(d[(1, 1)], c11, epsilon = 1e-3);
        assert_relative_eq!(d[(2, 2)], c11, epsilon = 1e-3);
        assert_relative_eq!(d[(3, 3)], c44, epsilon = 1e-3);

        // Check off-diagonal coupling
        assert_relative_eq!(d[(0, 1)], c12, epsilon = 1e-3);
        assert_relative_eq!(d[(0, 2)], c12, epsilon = 1e-3);
        assert_relative_eq!(d[(1, 2)], c12, epsilon = 1e-3);
    }
}
