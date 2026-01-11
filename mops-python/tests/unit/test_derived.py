"""Tests for derived quantity computation functions.

This module tests the stress and strain analysis functions in mops.derived:
- von Mises stress/strain
- Principal stresses/strains
- Tresca stress
- Maximum shear stress/strain
- Hydrostatic stress
- Deviatoric stress/strain
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mops.derived import (
    deviatoric_strain,
    deviatoric_stress,
    hydrostatic_stress,
    max_shear_strain,
    max_shear_stress,
    pressure,
    principal_strains,
    principal_stresses,
    stress_intensity,
    tresca_stress,
    volumetric_strain,
    von_mises_strain,
    von_mises_stress,
)


# =============================================================================
# Von Mises Stress Tests
# =============================================================================


class TestVonMisesStress:
    """Test von Mises stress computation."""

    def test_uniaxial_tension(self):
        """Von Mises of uniaxial tension σ_xx = σ equals σ."""
        # σ_xx = 100 MPa, all others zero
        stress = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        vm = von_mises_stress(stress)
        assert_allclose(vm, [100.0], rtol=1e-10)

    def test_pure_shear(self):
        """Von Mises of pure shear τ_xy = τ equals √3 * τ."""
        # τ_xy = 100 MPa
        stress = np.array([[0.0, 0.0, 0.0, 100.0, 0.0, 0.0]])
        vm = von_mises_stress(stress)
        assert_allclose(vm, [100.0 * np.sqrt(3)], rtol=1e-10)

    def test_hydrostatic_is_zero(self):
        """Von Mises of pure hydrostatic stress is zero."""
        # Equal triaxial: σ_xx = σ_yy = σ_zz = 100
        stress = np.array([[100.0, 100.0, 100.0, 0.0, 0.0, 0.0]])
        vm = von_mises_stress(stress)
        assert_allclose(vm, [0.0], atol=1e-10)

    def test_biaxial_stress(self):
        """Von Mises for biaxial stress state."""
        # σ_xx = 100, σ_yy = -100 (pure shear equivalent)
        stress = np.array([[100.0, -100.0, 0.0, 0.0, 0.0, 0.0]])
        vm = von_mises_stress(stress)
        # Expected: sqrt((100-(-100))^2 + 0 + 100^2 + 100^2)/sqrt(2) = sqrt(3)*100
        assert_allclose(vm, [100.0 * np.sqrt(3)], rtol=1e-10)

    def test_single_element(self):
        """Works with 1D input (single element)."""
        stress = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vm = von_mises_stress(stress)
        assert_allclose(vm, [100.0], rtol=1e-10)

    def test_vectorized(self):
        """Works with multiple elements."""
        stress = np.array([
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ])
        vm = von_mises_stress(stress)
        assert vm.shape == (3,)
        assert_allclose(vm, [100.0, 100.0, 100.0], rtol=1e-10)


# =============================================================================
# Principal Stresses Tests
# =============================================================================


class TestPrincipalStresses:
    """Test principal stress computation."""

    def test_uniaxial(self):
        """Principal stresses for uniaxial tension."""
        stress = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        principals = principal_stresses(stress)
        assert principals.shape == (1, 3)
        # σ1 = 100, σ2 = 0, σ3 = 0
        assert_allclose(principals[0], [100.0, 0.0, 0.0], atol=1e-10)

    def test_biaxial(self):
        """Principal stresses for biaxial stress state."""
        stress = np.array([[100.0, 50.0, 0.0, 0.0, 0.0, 0.0]])
        principals = principal_stresses(stress)
        # Already principal directions: σ1 = 100, σ2 = 50, σ3 = 0
        assert_allclose(principals[0], [100.0, 50.0, 0.0], atol=1e-10)

    def test_with_shear(self):
        """Principal stresses with shear component."""
        # 2D stress: σ_xx = 100, σ_yy = 0, τ_xy = 50
        # Mohr's circle: center = 50, radius = sqrt(50^2 + 50^2)
        stress = np.array([[100.0, 0.0, 0.0, 50.0, 0.0, 0.0]])
        principals = principal_stresses(stress)

        center = 50.0
        radius = np.sqrt(50.0**2 + 50.0**2)
        expected = [center + radius, 0.0, center - radius]
        assert_allclose(principals[0], expected, atol=1e-10)

    def test_ordering(self):
        """Principal stresses should be in descending order."""
        # Random stress state
        stress = np.array([[30.0, 50.0, 20.0, 10.0, 5.0, 8.0]])
        principals = principal_stresses(stress)
        # Check descending order
        assert principals[0, 0] >= principals[0, 1] >= principals[0, 2]


# =============================================================================
# Tresca Stress Tests
# =============================================================================


class TestTrescaStress:
    """Test Tresca (maximum shear) stress computation."""

    def test_uniaxial(self):
        """Tresca equals von Mises for uniaxial tension."""
        stress = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        tresca = tresca_stress(stress)
        vm = von_mises_stress(stress)
        assert_allclose(tresca, vm, rtol=1e-10)

    def test_pure_shear(self):
        """Tresca for pure shear: 2 * τ."""
        # σ_xx = 100, σ_yy = -100 is equivalent to pure shear
        # Principal: σ1 = 100, σ3 = -100, Tresca = 200
        stress = np.array([[100.0, -100.0, 0.0, 0.0, 0.0, 0.0]])
        tresca = tresca_stress(stress)
        assert_allclose(tresca, [200.0], rtol=1e-10)

    def test_biaxial(self):
        """Tresca for biaxial stress."""
        # σ1 = 100, σ2 = 50, σ3 = 0
        stress = np.array([[100.0, 50.0, 0.0, 0.0, 0.0, 0.0]])
        tresca = tresca_stress(stress)
        assert_allclose(tresca, [100.0], rtol=1e-10)


# =============================================================================
# Maximum Shear Stress Tests
# =============================================================================


class TestMaxShearStress:
    """Test maximum shear stress computation."""

    def test_uniaxial(self):
        """Max shear for uniaxial is σ/2."""
        stress = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        tau_max = max_shear_stress(stress)
        assert_allclose(tau_max, [50.0], rtol=1e-10)

    def test_relation_to_tresca(self):
        """Max shear is half of Tresca."""
        stress = np.array([[100.0, -100.0, 0.0, 0.0, 0.0, 0.0]])
        tau_max = max_shear_stress(stress)
        tresca = tresca_stress(stress)
        assert_allclose(tau_max, tresca / 2.0, rtol=1e-10)


# =============================================================================
# Hydrostatic Stress Tests
# =============================================================================


class TestHydrostaticStress:
    """Test hydrostatic stress computation."""

    def test_uniaxial(self):
        """Hydrostatic stress for uniaxial is σ/3."""
        stress = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        p_hydro = hydrostatic_stress(stress)
        assert_allclose(p_hydro, [100.0 / 3.0], rtol=1e-10)

    def test_triaxial(self):
        """Hydrostatic stress for triaxial is average of normal stresses."""
        stress = np.array([[100.0, 200.0, 300.0, 0.0, 0.0, 0.0]])
        p_hydro = hydrostatic_stress(stress)
        assert_allclose(p_hydro, [200.0], rtol=1e-10)

    def test_pressure_relation(self):
        """Pressure is negative hydrostatic stress."""
        stress = np.array([[100.0, 200.0, 300.0, 0.0, 0.0, 0.0]])
        p = pressure(stress)
        p_hydro = hydrostatic_stress(stress)
        assert_allclose(p, -p_hydro, rtol=1e-10)


# =============================================================================
# Stress Intensity Tests
# =============================================================================


class TestStressIntensity:
    """Test stress intensity computation."""

    def test_uniaxial(self):
        """Stress intensity for uniaxial equals principal stress."""
        stress = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        intensity = stress_intensity(stress)
        assert_allclose(intensity, [100.0], rtol=1e-10)

    def test_biaxial(self):
        """Stress intensity for biaxial."""
        # σ1 = 100, σ2 = 50, σ3 = 0
        # max(|100-50|, |50-0|, |0-100|) = 100
        stress = np.array([[100.0, 50.0, 0.0, 0.0, 0.0, 0.0]])
        intensity = stress_intensity(stress)
        assert_allclose(intensity, [100.0], rtol=1e-10)


# =============================================================================
# Deviatoric Stress Tests
# =============================================================================


class TestDeviatoricStress:
    """Test deviatoric stress computation."""

    def test_normal_components(self):
        """Normal deviatoric components should sum to zero."""
        stress = np.array([[100.0, 200.0, 300.0, 10.0, 20.0, 30.0]])
        dev = deviatoric_stress(stress)
        # Hydrostatic = 200, so dev = [-100, 0, 100, 10, 20, 30]
        assert_allclose(dev[0, 0], -100.0, rtol=1e-10)
        assert_allclose(dev[0, 1], 0.0, atol=1e-10)
        assert_allclose(dev[0, 2], 100.0, rtol=1e-10)

    def test_shear_components_unchanged(self):
        """Shear components should be unchanged in deviatoric."""
        stress = np.array([[100.0, 200.0, 300.0, 10.0, 20.0, 30.0]])
        dev = deviatoric_stress(stress)
        assert_allclose(dev[0, 3:], stress[0, 3:], rtol=1e-10)


# =============================================================================
# Von Mises Strain Tests
# =============================================================================


class TestVonMisesStrain:
    """Test von Mises equivalent strain computation."""

    def test_uniaxial(self):
        """Equivalent strain for uniaxial strain."""
        # Pure axial strain ε_xx = 0.001
        strain = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
        eq = von_mises_strain(strain)
        # Expected: sqrt(2/3) * sqrt(2*ε²)/sqrt(2) = sqrt(2/3) * ε
        expected = 0.001 * np.sqrt(2.0 / 3.0)
        assert_allclose(eq, [expected], rtol=1e-10)


# =============================================================================
# Principal Strains Tests
# =============================================================================


class TestPrincipalStrains:
    """Test principal strain computation."""

    def test_uniaxial(self):
        """Principal strains for uniaxial strain."""
        strain = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
        principals = principal_strains(strain)
        assert_allclose(principals[0], [0.001, 0.0, 0.0], atol=1e-15)

    def test_ordering(self):
        """Principal strains should be in descending order."""
        strain = np.array([[0.001, 0.002, 0.0005, 0.0001, 0.0, 0.0]])
        principals = principal_strains(strain)
        assert principals[0, 0] >= principals[0, 1] >= principals[0, 2]


# =============================================================================
# Maximum Shear Strain Tests
# =============================================================================


class TestMaxShearStrain:
    """Test maximum shear strain computation."""

    def test_uniaxial(self):
        """Max shear strain for uniaxial is ε1 - ε3."""
        strain = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
        gamma_max = max_shear_strain(strain)
        assert_allclose(gamma_max, [0.001], rtol=1e-10)


# =============================================================================
# Volumetric Strain Tests
# =============================================================================


class TestVolumetricStrain:
    """Test volumetric strain computation."""

    def test_sum_of_normal(self):
        """Volumetric strain is sum of normal strains."""
        strain = np.array([[0.001, 0.002, 0.003, 0.0, 0.0, 0.0]])
        vol = volumetric_strain(strain)
        assert_allclose(vol, [0.006], rtol=1e-10)


# =============================================================================
# Deviatoric Strain Tests
# =============================================================================


class TestDeviatoricStrain:
    """Test deviatoric strain computation."""

    def test_normal_components(self):
        """Normal deviatoric components should sum to zero."""
        strain = np.array([[0.003, 0.002, 0.001, 0.0, 0.0, 0.0]])
        dev = deviatoric_strain(strain)
        # Mean = 0.002, so dev = [0.001, 0, -0.001, 0, 0, 0]
        assert_allclose(dev[0, :3], [0.001, 0.0, -0.001], atol=1e-15)

    def test_shear_components_unchanged(self):
        """Shear components should be unchanged in deviatoric."""
        strain = np.array([[0.003, 0.002, 0.001, 0.0001, 0.0002, 0.0003]])
        dev = deviatoric_strain(strain)
        assert_allclose(dev[0, 3:], strain[0, 3:], rtol=1e-10)
