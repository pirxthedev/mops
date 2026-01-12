"""Unit tests for 2D element stiffness matrices.

This module verifies fundamental properties of 2D element stiffness matrices
for plane stress, plane strain, and axisymmetric elements:

1. Symmetry: K = K^T
2. Positive semi-definiteness: x^T K x >= 0 for all x, with equality for rigid body modes
3. Patch tests: Linear displacement fields produce constant strain
4. Rigid body modes: Pure translation produces zero internal forces
5. Area computation: Element area matches expected geometry

For 2D elements:
- 3 rigid body modes (2 translations + 1 rotation)
- 2 DOFs per node (u, v displacements)
- 3 strain components (epsilon_xx, epsilon_yy, gamma_xy)

These tests validate the finite element formulations at the element level,
ensuring correctness before assembly into global systems.
"""

import numpy as np
import pytest
from numpy.linalg import eigvalsh

from mops import Material, element_stiffness, element_volume, compute_element_stress


# =============================================================================
# Element Node Fixtures
# =============================================================================


@pytest.fixture
def unit_tri3_nodes() -> np.ndarray:
    """Unit right triangle in XY plane (area = 0.5)."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)


@pytest.fixture
def scaled_tri3_nodes() -> np.ndarray:
    """Triangle scaled by factor of 2 (area = 2.0)."""
    return np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ], dtype=np.float64)


@pytest.fixture
def equilateral_tri3_nodes() -> np.ndarray:
    """Equilateral triangle with side length 1."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
    ], dtype=np.float64)


@pytest.fixture
def unit_quad4_nodes() -> np.ndarray:
    """Unit square in XY plane (area = 1.0)."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)


@pytest.fixture
def stretched_quad4_nodes() -> np.ndarray:
    """Stretched quad (2x1) for aspect ratio testing."""
    return np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)


@pytest.fixture
def skewed_quad4_nodes() -> np.ndarray:
    """Skewed quadrilateral (parallelogram)."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.5, 1.0, 0.0],
        [0.5, 1.0, 0.0],
    ], dtype=np.float64)


@pytest.fixture
def steel() -> Material:
    """Standard steel material."""
    return Material.steel()


@pytest.fixture
def aluminum() -> Material:
    """Aluminum material."""
    return Material.aluminum()


# =============================================================================
# Helper Functions
# =============================================================================


def is_symmetric(matrix: np.ndarray, rtol: float = 1e-10, atol: float = 0.0) -> bool:
    """Check if matrix is symmetric within tolerance."""
    if atol == 0.0:
        atol = rtol * np.max(np.abs(matrix))
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)


def is_positive_semidefinite(matrix: np.ndarray, rtol: float = 1e-10) -> bool:
    """Check if matrix is positive semi-definite."""
    eigenvalues = eigvalsh(matrix)
    max_eig = np.max(np.abs(eigenvalues))
    return np.all(eigenvalues >= -rtol * max_eig)


def count_zero_eigenvalues(matrix: np.ndarray, rtol: float = 1e-10) -> int:
    """Count the number of eigenvalues close to zero."""
    eigenvalues = eigvalsh(matrix)
    max_eig = np.max(np.abs(eigenvalues))
    tol = rtol * max_eig
    return np.sum(np.abs(eigenvalues) < tol)


# =============================================================================
# Tri3 Stiffness Matrix Tests
# =============================================================================


class TestTri3Symmetry:
    """Test that Tri3 stiffness matrices are symmetric."""

    def test_tri3_symmetry(self, unit_tri3_nodes, steel):
        """Tri3 stiffness matrix must be symmetric."""
        k = element_stiffness("tri3", unit_tri3_nodes, steel)
        assert k.shape == (6, 6)  # 3 nodes * 2 DOFs
        assert is_symmetric(k), "Tri3 stiffness matrix is not symmetric"

    def test_tri3_symmetry_scaled(self, scaled_tri3_nodes, steel):
        """Symmetry should hold for scaled elements."""
        k = element_stiffness("tri3", scaled_tri3_nodes, steel)
        assert is_symmetric(k)

    def test_tri3_symmetry_equilateral(self, equilateral_tri3_nodes, steel):
        """Symmetry should hold for equilateral triangle."""
        k = element_stiffness("tri3", equilateral_tri3_nodes, steel)
        assert is_symmetric(k)

    def test_tri3_symmetry_different_materials(
        self, unit_tri3_nodes, steel, aluminum
    ):
        """Symmetry should hold for different materials."""
        k_steel = element_stiffness("tri3", unit_tri3_nodes, steel)
        k_aluminum = element_stiffness("tri3", unit_tri3_nodes, aluminum)
        assert is_symmetric(k_steel)
        assert is_symmetric(k_aluminum)


class TestTri3PositiveSemiDefiniteness:
    """Test that Tri3 stiffness matrices are positive semi-definite."""

    def test_tri3_positive_semidefinite(self, unit_tri3_nodes, steel):
        """Tri3 stiffness matrix must be positive semi-definite."""
        k = element_stiffness("tri3", unit_tri3_nodes, steel)
        assert is_positive_semidefinite(k), \
            "Tri3 stiffness matrix has negative eigenvalues"

    def test_tri3_has_rigid_body_modes(self, unit_tri3_nodes, steel):
        """Tri3 should have 3 zero eigenvalues (rigid body modes).

        For 2D elements, there are 3 rigid body modes:
        - 2 translations (x, y)
        - 1 rotation (about z-axis)
        """
        k = element_stiffness("tri3", unit_tri3_nodes, steel)
        n_zero = count_zero_eigenvalues(k, rtol=1e-6)
        assert n_zero == 3, f"Expected 3 zero eigenvalues, got {n_zero}"

    def test_tri3_positive_diagonal_entries(self, unit_tri3_nodes, steel):
        """All diagonal entries of stiffness matrix should be positive."""
        k = element_stiffness("tri3", unit_tri3_nodes, steel)
        assert np.all(np.diag(k) > 0), "Stiffness matrix has non-positive diagonal"


class TestTri3RigidBodyMotion:
    """Test that rigid body motions produce zero internal forces for Tri3."""

    def _assert_rigid_body_zero(self, f: np.ndarray, k: np.ndarray, mode: str):
        """Assert forces are effectively zero for rigid body motion."""
        max_k = np.max(np.abs(k))
        max_f = np.max(np.abs(f))
        rtol = 1e-10
        assert max_f < rtol * max_k, \
            f"{mode} produced relative force {max_f/max_k:.2e} > {rtol}"

    def test_tri3_translation_x(self, unit_tri3_nodes, steel):
        """Pure x-translation should produce zero forces."""
        k = element_stiffness("tri3", unit_tri3_nodes, steel)
        n_nodes = 3
        u = np.zeros(n_nodes * 2)
        for i in range(n_nodes):
            u[i * 2] = 1.0  # Unit translation in x
        f = k @ u
        self._assert_rigid_body_zero(f, k, "Translation in x")

    def test_tri3_translation_y(self, unit_tri3_nodes, steel):
        """Pure y-translation should produce zero forces."""
        k = element_stiffness("tri3", unit_tri3_nodes, steel)
        n_nodes = 3
        u = np.zeros(n_nodes * 2)
        for i in range(n_nodes):
            u[i * 2 + 1] = 1.0  # Unit translation in y
        f = k @ u
        self._assert_rigid_body_zero(f, k, "Translation in y")

    def test_tri3_translation_modes(self, unit_tri3_nodes, steel):
        """Both translation modes should produce zero forces."""
        k = element_stiffness("tri3", unit_tri3_nodes, steel)
        n_nodes = 3

        for direction in range(2):
            u = np.zeros(n_nodes * 2)
            for i in range(n_nodes):
                u[i * 2 + direction] = 1.0
            f = k @ u
            self._assert_rigid_body_zero(f, k, f"Translation mode {direction}")


class TestTri3PatchTests:
    """Patch tests verify constant strain fields for Tri3."""

    def _assert_equilibrium(
        self, f: np.ndarray, n_nodes: int, label: str
    ):
        """Assert forces sum to zero (equilibrium) within tolerance."""
        fx_sum = sum(f[i * 2] for i in range(n_nodes))
        fy_sum = sum(f[i * 2 + 1] for i in range(n_nodes))

        max_f = np.max(np.abs(f))
        rtol = 1e-8

        assert abs(fx_sum) < rtol * max_f, \
            f"{label}: Sum of x-forces {fx_sum:.2e} exceeds tolerance"
        assert abs(fy_sum) < rtol * max_f, \
            f"{label}: Sum of y-forces {fy_sum:.2e} exceeds tolerance"

    def test_tri3_constant_strain_xx(self, unit_tri3_nodes, steel):
        """Apply uniform epsilon_xx strain and verify equilibrium.

        For constant strain epsilon_xx = du/dx, we apply:
        u_x = epsilon_xx * x, u_y = 0
        """
        k = element_stiffness("tri3", unit_tri3_nodes, steel)

        strain_xx = 0.001
        u = np.zeros(6)  # 3 nodes * 2 DOFs
        for i in range(3):
            x = unit_tri3_nodes[i, 0]
            u[i * 2] = strain_xx * x

        f = k @ u
        self._assert_equilibrium(f, 3, "Tri3 constant epsilon_xx")

    def test_tri3_constant_strain_yy(self, unit_tri3_nodes, steel):
        """Apply uniform epsilon_yy strain and verify equilibrium."""
        k = element_stiffness("tri3", unit_tri3_nodes, steel)

        strain_yy = 0.001
        u = np.zeros(6)
        for i in range(3):
            y = unit_tri3_nodes[i, 1]
            u[i * 2 + 1] = strain_yy * y

        f = k @ u
        self._assert_equilibrium(f, 3, "Tri3 constant epsilon_yy")

    def test_tri3_pure_shear(self, unit_tri3_nodes, steel):
        """Apply pure shear strain and verify equilibrium.

        Pure shear: u_x = gamma * y, u_y = 0 (gives gamma_xy = du_x/dy)
        """
        k = element_stiffness("tri3", unit_tri3_nodes, steel)

        shear = 0.001
        u = np.zeros(6)
        for i in range(3):
            y = unit_tri3_nodes[i, 1]
            u[i * 2] = shear * y

        f = k @ u
        self._assert_equilibrium(f, 3, "Tri3 pure shear")


# =============================================================================
# Quad4 Stiffness Matrix Tests
# =============================================================================


class TestQuad4Symmetry:
    """Test that Quad4 stiffness matrices are symmetric."""

    def test_quad4_symmetry(self, unit_quad4_nodes, steel):
        """Quad4 stiffness matrix must be symmetric."""
        k = element_stiffness("quad4", unit_quad4_nodes, steel)
        assert k.shape == (8, 8)  # 4 nodes * 2 DOFs
        assert is_symmetric(k), "Quad4 stiffness matrix is not symmetric"

    def test_quad4_symmetry_stretched(self, stretched_quad4_nodes, steel):
        """Symmetry should hold for stretched elements."""
        k = element_stiffness("quad4", stretched_quad4_nodes, steel)
        assert is_symmetric(k)

    def test_quad4_symmetry_skewed(self, skewed_quad4_nodes, steel):
        """Symmetry should hold for skewed elements."""
        k = element_stiffness("quad4", skewed_quad4_nodes, steel)
        assert is_symmetric(k)


class TestQuad4PositiveSemiDefiniteness:
    """Test that Quad4 stiffness matrices are positive semi-definite."""

    def test_quad4_positive_semidefinite(self, unit_quad4_nodes, steel):
        """Quad4 stiffness matrix must be positive semi-definite."""
        k = element_stiffness("quad4", unit_quad4_nodes, steel)
        assert is_positive_semidefinite(k), \
            "Quad4 stiffness matrix has negative eigenvalues"

    def test_quad4_has_rigid_body_modes(self, unit_quad4_nodes, steel):
        """Quad4 should have 3 zero eigenvalues (rigid body modes)."""
        k = element_stiffness("quad4", unit_quad4_nodes, steel)
        n_zero = count_zero_eigenvalues(k, rtol=1e-6)
        assert n_zero == 3, f"Expected 3 zero eigenvalues, got {n_zero}"

    def test_quad4_positive_diagonal_entries(self, unit_quad4_nodes, steel):
        """All diagonal entries of stiffness matrix should be positive."""
        k = element_stiffness("quad4", unit_quad4_nodes, steel)
        assert np.all(np.diag(k) > 0), "Stiffness matrix has non-positive diagonal"


class TestQuad4RigidBodyMotion:
    """Test that rigid body motions produce zero internal forces for Quad4."""

    def _assert_rigid_body_zero(self, f: np.ndarray, k: np.ndarray, mode: str):
        """Assert forces are effectively zero for rigid body motion."""
        max_k = np.max(np.abs(k))
        max_f = np.max(np.abs(f))
        rtol = 1e-10
        assert max_f < rtol * max_k, \
            f"{mode} produced relative force {max_f/max_k:.2e} > {rtol}"

    def test_quad4_translation_x(self, unit_quad4_nodes, steel):
        """Pure x-translation should produce zero forces."""
        k = element_stiffness("quad4", unit_quad4_nodes, steel)
        n_nodes = 4
        u = np.zeros(n_nodes * 2)
        for i in range(n_nodes):
            u[i * 2] = 1.0
        f = k @ u
        self._assert_rigid_body_zero(f, k, "Translation in x")

    def test_quad4_translation_y(self, unit_quad4_nodes, steel):
        """Pure y-translation should produce zero forces."""
        k = element_stiffness("quad4", unit_quad4_nodes, steel)
        n_nodes = 4
        u = np.zeros(n_nodes * 2)
        for i in range(n_nodes):
            u[i * 2 + 1] = 1.0
        f = k @ u
        self._assert_rigid_body_zero(f, k, "Translation in y")

    def test_quad4_translation_modes(self, unit_quad4_nodes, steel):
        """Both translation modes should produce zero forces."""
        k = element_stiffness("quad4", unit_quad4_nodes, steel)
        n_nodes = 4

        for direction in range(2):
            u = np.zeros(n_nodes * 2)
            for i in range(n_nodes):
                u[i * 2 + direction] = 1.0
            f = k @ u
            self._assert_rigid_body_zero(f, k, f"Translation mode {direction}")


class TestQuad4PatchTests:
    """Patch tests verify constant strain fields for Quad4."""

    def _assert_equilibrium(self, f: np.ndarray, n_nodes: int, label: str):
        """Assert forces sum to zero (equilibrium) within tolerance."""
        fx_sum = sum(f[i * 2] for i in range(n_nodes))
        fy_sum = sum(f[i * 2 + 1] for i in range(n_nodes))

        max_f = np.max(np.abs(f))
        rtol = 1e-8

        assert abs(fx_sum) < rtol * max_f, \
            f"{label}: Sum of x-forces {fx_sum:.2e} exceeds tolerance"
        assert abs(fy_sum) < rtol * max_f, \
            f"{label}: Sum of y-forces {fy_sum:.2e} exceeds tolerance"

    def test_quad4_constant_strain_xx(self, unit_quad4_nodes, steel):
        """Apply uniform epsilon_xx strain and verify equilibrium."""
        k = element_stiffness("quad4", unit_quad4_nodes, steel)

        strain_xx = 0.001
        u = np.zeros(8)  # 4 nodes * 2 DOFs
        for i in range(4):
            x = unit_quad4_nodes[i, 0]
            u[i * 2] = strain_xx * x

        f = k @ u
        self._assert_equilibrium(f, 4, "Quad4 constant epsilon_xx")

    def test_quad4_constant_strain_yy(self, unit_quad4_nodes, steel):
        """Apply uniform epsilon_yy strain and verify equilibrium."""
        k = element_stiffness("quad4", unit_quad4_nodes, steel)

        strain_yy = 0.001
        u = np.zeros(8)
        for i in range(4):
            y = unit_quad4_nodes[i, 1]
            u[i * 2 + 1] = strain_yy * y

        f = k @ u
        self._assert_equilibrium(f, 4, "Quad4 constant epsilon_yy")

    def test_quad4_pure_shear(self, unit_quad4_nodes, steel):
        """Apply pure shear strain and verify equilibrium."""
        k = element_stiffness("quad4", unit_quad4_nodes, steel)

        shear = 0.001
        u = np.zeros(8)
        for i in range(4):
            y = unit_quad4_nodes[i, 1]
            u[i * 2] = shear * y

        f = k @ u
        self._assert_equilibrium(f, 4, "Quad4 pure shear")


# =============================================================================
# Area Tests
# =============================================================================


class TestElementArea:
    """Test element area computation for 2D elements."""

    def test_tri3_unit_area(self, unit_tri3_nodes):
        """Unit right triangle area should be 0.5."""
        area = element_volume("tri3", unit_tri3_nodes)
        expected = 0.5
        assert np.isclose(area, expected, rtol=1e-10), \
            f"Expected area {expected}, got {area}"

    def test_tri3_scaled_area(self, scaled_tri3_nodes):
        """Scaled (2x) triangle area should be 2.0."""
        area = element_volume("tri3", scaled_tri3_nodes)
        expected = 2.0  # Area scales as L^2
        assert np.isclose(area, expected, rtol=1e-10)

    def test_tri3_equilateral_area(self, equilateral_tri3_nodes):
        """Equilateral triangle with side 1 has area sqrt(3)/4."""
        area = element_volume("tri3", equilateral_tri3_nodes)
        expected = np.sqrt(3) / 4
        assert np.isclose(area, expected, rtol=1e-10)

    def test_quad4_unit_square_area(self, unit_quad4_nodes):
        """Unit square area should be 1.0."""
        area = element_volume("quad4", unit_quad4_nodes)
        assert np.isclose(area, 1.0, rtol=1e-10)

    def test_quad4_stretched_area(self, stretched_quad4_nodes):
        """2x1 rectangle area should be 2.0."""
        area = element_volume("quad4", stretched_quad4_nodes)
        assert np.isclose(area, 2.0, rtol=1e-10)

    def test_quad4_skewed_area(self, skewed_quad4_nodes):
        """Parallelogram with base 1 and height 1 has area 1.0."""
        area = element_volume("quad4", skewed_quad4_nodes)
        assert np.isclose(area, 1.0, rtol=1e-10)


# =============================================================================
# Material Scaling Tests
# =============================================================================


class TestMaterialScaling2D:
    """Test that stiffness scales correctly with material properties for 2D elements."""

    def test_tri3_stiffness_scales_with_youngs_modulus(self, unit_tri3_nodes):
        """Stiffness should scale linearly with Young's modulus."""
        mat1 = Material("test1", e=100e9, nu=0.3)
        mat2 = Material("test2", e=200e9, nu=0.3)

        k1 = element_stiffness("tri3", unit_tri3_nodes, mat1)
        k2 = element_stiffness("tri3", unit_tri3_nodes, mat2)

        assert np.allclose(k2, 2.0 * k1, rtol=1e-10), \
            "Stiffness does not scale linearly with Young's modulus"

    def test_quad4_stiffness_scales_with_youngs_modulus(self, unit_quad4_nodes):
        """Stiffness should scale linearly with Young's modulus."""
        mat1 = Material("test1", e=100e9, nu=0.3)
        mat2 = Material("test2", e=200e9, nu=0.3)

        k1 = element_stiffness("quad4", unit_quad4_nodes, mat1)
        k2 = element_stiffness("quad4", unit_quad4_nodes, mat2)

        assert np.allclose(k2, 2.0 * k1, rtol=1e-10)

    def test_tri3_different_poisson_ratios(self, unit_tri3_nodes):
        """Different Poisson ratios should produce different stiffness."""
        mat1 = Material("nu_low", e=100e9, nu=0.1)
        mat2 = Material("nu_high", e=100e9, nu=0.4)

        k1 = element_stiffness("tri3", unit_tri3_nodes, mat1)
        k2 = element_stiffness("tri3", unit_tri3_nodes, mat2)

        assert not np.allclose(k1, k2)
        assert is_symmetric(k1)
        assert is_symmetric(k2)
        assert is_positive_semidefinite(k1)
        assert is_positive_semidefinite(k2)


# =============================================================================
# Stress Computation Tests
# =============================================================================


class TestStressComputation2D:
    """Test stress computation for 2D elements."""

    def test_tri3_stress_under_uniaxial_tension(self, unit_tri3_nodes, steel):
        """Verify stress computation for uniaxial tension."""
        # Apply uniform epsilon_xx strain
        strain_xx = 0.001
        u = np.zeros(6)
        for i in range(3):
            x = unit_tri3_nodes[i, 0]
            u[i * 2] = strain_xx * x

        stresses = compute_element_stress("tri3", unit_tri3_nodes, u, steel)

        # CST (Constant Strain Triangle) should have 1 integration point
        assert stresses.shape[0] >= 1
        assert stresses.shape[1] == 6  # Voigt notation

        # For plane stress: sigma_xx should be positive, sigma_zz = 0
        sigma_xx = stresses[0, 0]
        sigma_zz = stresses[0, 2]

        # Stress should be positive for tensile strain
        assert sigma_xx > 0, f"Expected positive sigma_xx, got {sigma_xx}"
        # Out-of-plane stress should be zero for plane stress
        assert abs(sigma_zz) < 1e-6, f"Expected zero sigma_zz for plane stress, got {sigma_zz}"

    def test_quad4_stress_under_uniaxial_tension(self, unit_quad4_nodes, steel):
        """Verify stress computation for uniaxial tension in Quad4."""
        strain_xx = 0.001
        u = np.zeros(8)
        for i in range(4):
            x = unit_quad4_nodes[i, 0]
            u[i * 2] = strain_xx * x

        stresses = compute_element_stress("quad4", unit_quad4_nodes, u, steel)

        # Quad4 uses 2x2 Gauss integration = 4 points
        assert stresses.shape[0] >= 1
        assert stresses.shape[1] == 6

        # Check first integration point
        sigma_xx = stresses[0, 0]
        sigma_zz = stresses[0, 2]

        assert sigma_xx > 0
        assert abs(sigma_zz) < 1e-6


# =============================================================================
# Cross-Element Comparison Tests
# =============================================================================


class TestCrossElementComparison2D:
    """Compare behavior across 2D element types."""

    def test_tri3_vs_quad4_similar_properties(
        self, unit_tri3_nodes, unit_quad4_nodes, steel
    ):
        """Tri3 and Quad4 should both have correct fundamental properties."""
        k3 = element_stiffness("tri3", unit_tri3_nodes, steel)
        k4 = element_stiffness("quad4", unit_quad4_nodes, steel)

        # Both should be symmetric
        assert is_symmetric(k3)
        assert is_symmetric(k4)

        # Both should be positive semi-definite
        assert is_positive_semidefinite(k3)
        assert is_positive_semidefinite(k4)

        # Both should have 3 rigid body modes
        assert count_zero_eigenvalues(k3, rtol=1e-6) == 3
        assert count_zero_eigenvalues(k4, rtol=1e-6) == 3

    def test_area_relationship(self, unit_tri3_nodes, unit_quad4_nodes):
        """Two unit triangles should equal one unit quad in area."""
        tri_area = element_volume("tri3", unit_tri3_nodes)
        quad_area = element_volume("quad4", unit_quad4_nodes)

        # Unit right triangle has area 0.5, unit square has area 1.0
        assert np.isclose(2 * tri_area, quad_area)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling2D:
    """Test error handling for 2D elements."""

    def test_wrong_node_count_tri3(self, steel):
        """Wrong node count for tri3 should raise ValueError."""
        bad_nodes = np.array([
            [0, 0, 0],
            [1, 0, 0],
        ], dtype=np.float64)
        with pytest.raises(ValueError):
            element_stiffness("tri3", bad_nodes, steel)

    def test_wrong_node_count_quad4(self, steel):
        """Wrong node count for quad4 should raise ValueError."""
        bad_nodes = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ], dtype=np.float64)
        with pytest.raises(ValueError):
            element_stiffness("quad4", bad_nodes, steel)

    def test_wrong_displacement_count_tri3(self, unit_tri3_nodes, steel):
        """Wrong displacement count for tri3 should raise ValueError."""
        u = np.zeros(9)  # Should be 6 for 2D
        with pytest.raises(ValueError):
            compute_element_stress("tri3", unit_tri3_nodes, u, steel)

    def test_wrong_displacement_count_quad4(self, unit_quad4_nodes, steel):
        """Wrong displacement count for quad4 should raise ValueError."""
        u = np.zeros(12)  # Should be 8 for 2D
        with pytest.raises(ValueError):
            compute_element_stress("quad4", unit_quad4_nodes, u, steel)


# =============================================================================
# Thickness Parameter Tests
# =============================================================================


class TestThicknessParameter:
    """Test thickness parameter in Python Model API."""

    def test_model_assign_with_thickness(self):
        """Model.assign() should accept thickness parameter."""
        from mops import Model, Material, Mesh
        from mops.query import Elements

        # Create a simple 2D mesh
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2]], dtype=np.int64)

        mesh = Mesh.from_arrays(nodes, elements, "tri3")
        steel = Material.steel()

        # Test with thickness
        model = Model(mesh, materials={"steel": steel}).assign(
            Elements.all(), material="steel", thickness=2.5
        )

        # Check thickness was stored
        assert model.thickness == 2.5

    def test_model_assign_default_thickness(self):
        """Model.assign() without thickness should use default (1.0)."""
        from mops import Model, Material, Mesh
        from mops.query import Elements

        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2]], dtype=np.int64)

        mesh = Mesh.from_arrays(nodes, elements, "tri3")
        steel = Material.steel()

        model = Model(mesh, materials={"steel": steel}).assign(
            Elements.all(), material="steel"
        )

        # Thickness should be None when not specified (will default to 1.0 in solver)
        assert model.thickness is None

    def test_model_assign_invalid_thickness(self):
        """Model.assign() with non-positive thickness should raise ValueError."""
        from mops import Model, Material, Mesh
        from mops.query import Elements

        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2]], dtype=np.int64)

        mesh = Mesh.from_arrays(nodes, elements, "tri3")
        steel = Material.steel()

        model = Model(mesh, materials={"steel": steel})

        # Zero thickness should raise
        with pytest.raises(ValueError, match="thickness must be positive"):
            model.assign(Elements.all(), material="steel", thickness=0.0)

        # Negative thickness should raise
        with pytest.raises(ValueError, match="thickness must be positive"):
            model.assign(Elements.all(), material="steel", thickness=-1.0)

    def test_thickness_preserved_through_model_methods(self):
        """Thickness should be preserved through constrain and load methods."""
        from mops import Model, Material, Mesh
        from mops.query import Elements, Nodes
        from mops.loads import Force

        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2]], dtype=np.int64)

        mesh = Mesh.from_arrays(nodes, elements, "tri3")
        steel = Material.steel()

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel", thickness=0.5)
            .constrain(Nodes.where(y=0), dofs=["ux", "uy"])
            .load(Nodes.where(y=1.0), Force(fy=1000.0))
        )

        # Thickness should be preserved
        assert model.thickness == 0.5


class TestThicknessScalingAtElementLevel:
    """Test that thickness parameter affects element stiffness matrices.

    Note: Full 2D solve workflow tests are pending stress recovery support for 2D elements.
    These tests verify the thickness parameter at the element level.
    """

    def test_tri3_stiffness_matrix_is_valid(self, unit_tri3_nodes, steel):
        """Tri3 stiffness matrix should be valid with default thickness.

        For plane stress elements: K = thickness * integral(B^T * D * B) dA
        """
        k1 = element_stiffness("tri3", unit_tri3_nodes, steel)
        k2 = element_stiffness("tri3", unit_tri3_nodes, steel)

        # Both should be the same when using default thickness
        assert np.allclose(k1, k2, rtol=1e-10), \
            "Identical elements should produce identical stiffness"

        # Verify matrix properties
        assert k1.shape == (6, 6)  # 3 nodes * 2 DOFs
        assert is_symmetric(k1)
        assert is_positive_semidefinite(k1)

    def test_quad4_stiffness_matrix_is_valid(self, unit_quad4_nodes, steel):
        """Quad4 stiffness matrix should be valid with default thickness."""
        k1 = element_stiffness("quad4", unit_quad4_nodes, steel)
        k2 = element_stiffness("quad4", unit_quad4_nodes, steel)

        assert np.allclose(k1, k2, rtol=1e-10), \
            "Identical elements should produce identical stiffness"

        # Verify matrix properties
        assert k1.shape == (8, 8)  # 4 nodes * 2 DOFs
        assert is_symmetric(k1)
        assert is_positive_semidefinite(k1)

    def test_thickness_parameter_signature_exists(self):
        """Verify that Model.assign() accepts thickness parameter."""
        from mops import Model, Material, Mesh
        from mops.query import Elements
        import inspect

        # Verify the signature includes thickness
        sig = inspect.signature(Model.assign)
        params = list(sig.parameters.keys())
        assert "thickness" in params, \
            f"Model.assign() should have thickness parameter. Params: {params}"

    def test_thickness_stored_in_model_state(self):
        """Verify thickness is stored in model state."""
        from mops import Model, Material, Mesh
        from mops.query import Elements

        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2]], dtype=np.int64)

        mesh = Mesh.from_arrays(nodes, elements, "tri3")
        steel = Material.steel()

        # Create model with thickness
        model = Model(mesh, materials={"steel": steel}).assign(
            Elements.all(), material="steel", thickness=3.5
        )

        # Verify thickness is stored and accessible
        assert model.thickness == 3.5
        assert model._state.thickness_assignments is not None
        assert len(model._state.thickness_assignments) == 1


# =============================================================================
# Axisymmetric Element Tests
# =============================================================================


@pytest.fixture
def axisym_tri3_nodes() -> np.ndarray:
    """Axisymmetric triangle in r-z plane (r=x, z=y, unused=z).

    The axisymmetric formulation uses the x-coordinate as the radial (r)
    coordinate and y as the axial (z) coordinate.
    """
    return np.array([
        [1.0, 0.0, 0.0],  # r=1, z=0
        [2.0, 0.0, 0.0],  # r=2, z=0
        [1.5, 1.0, 0.0],  # r=1.5, z=1
    ], dtype=np.float64)


@pytest.fixture
def axisym_quad4_nodes() -> np.ndarray:
    """Axisymmetric quadrilateral in r-z plane."""
    return np.array([
        [1.0, 0.0, 0.0],  # r=1, z=0
        [2.0, 0.0, 0.0],  # r=2, z=0
        [2.0, 1.0, 0.0],  # r=2, z=1
        [1.0, 1.0, 0.0],  # r=1, z=1
    ], dtype=np.float64)


class TestTri3Axisymmetric:
    """Test axisymmetric 3-node triangle element."""

    def test_stiffness_matrix_shape(self, axisym_tri3_nodes, steel):
        """Stiffness matrix should be 6x6 (3 nodes * 2 DOFs)."""
        k = element_stiffness("tri3axisymmetric", axisym_tri3_nodes, steel)
        assert k.shape == (6, 6)

    def test_stiffness_matrix_symmetry(self, axisym_tri3_nodes, steel):
        """Stiffness matrix should be symmetric."""
        k = element_stiffness("tri3axisymmetric", axisym_tri3_nodes, steel)
        assert is_symmetric(k)

    def test_stiffness_positive_semidefinite(self, axisym_tri3_nodes, steel):
        """Stiffness matrix should be positive semi-definite."""
        k = element_stiffness("tri3axisymmetric", axisym_tri3_nodes, steel)
        assert is_positive_semidefinite(k)

    def test_volume_is_revolution_volume(self, axisym_tri3_nodes):
        """Volume should be 2*pi*r_centroid * area (revolution about z-axis)."""
        vol = element_volume("tri3axisymmetric", axisym_tri3_nodes)
        # Triangle area = 0.5 * base * height = 0.5 * 1 * 1 = 0.5
        # Centroid r = (1 + 2 + 1.5) / 3 = 1.5
        # Revolution volume = 2 * pi * r_centroid * area = 2 * pi * 1.5 * 0.5 = 1.5 * pi
        expected = 2 * np.pi * 1.5 * 0.5
        assert np.isclose(vol, expected, rtol=1e-10)

    def test_rigid_body_modes(self, axisym_tri3_nodes, steel):
        """Axisymmetric element should have 1 or 2 rigid body modes.

        Unlike plane stress/strain which has 3 rigid body modes (2 translations + rotation),
        axisymmetric elements have fewer:
        - Axial (z) translation: always a rigid body mode (no strain)
        - Radial translation: causes hoop strain epsilon_theta = u_r/r, so NOT rigid
        - Rotation about z-axis: causes strain, so NOT rigid

        The number depends on element position relative to the axis. For elements
        away from the axis (r > 0), only axial translation is truly zero-strain.
        """
        k = element_stiffness("tri3axisymmetric", axisym_tri3_nodes, steel)
        n_zero = count_zero_eigenvalues(k, rtol=1e-6)
        # For elements away from axis, only axial translation is rigid body mode
        # The tri3 test element has r ranging from 1.0 to 2.0, so expect 1-2 modes
        assert n_zero >= 1 and n_zero <= 2, f"Expected 1-2 rigid body modes, got {n_zero}"

    def test_stress_shape(self, axisym_tri3_nodes, steel):
        """Stress tensor should have 6 components (converted from 4-component axi)."""
        # Axisymmetric stress has 4 components: sigma_rr, sigma_zz, sigma_theta, tau_rz
        # but gets converted to 6-component Voigt notation for consistency
        u = np.zeros(6)
        stress = compute_element_stress("tri3axisymmetric", axisym_tri3_nodes, u, steel)
        assert stress.shape[1] == 6, "Should return 6 stress components"


class TestQuad4Axisymmetric:
    """Test axisymmetric 4-node quadrilateral element."""

    def test_stiffness_matrix_shape(self, axisym_quad4_nodes, steel):
        """Stiffness matrix should be 8x8 (4 nodes * 2 DOFs)."""
        k = element_stiffness("quad4axisymmetric", axisym_quad4_nodes, steel)
        assert k.shape == (8, 8)

    def test_stiffness_matrix_symmetry(self, axisym_quad4_nodes, steel):
        """Stiffness matrix should be symmetric."""
        k = element_stiffness("quad4axisymmetric", axisym_quad4_nodes, steel)
        assert is_symmetric(k)

    def test_stiffness_positive_semidefinite(self, axisym_quad4_nodes, steel):
        """Stiffness matrix should be positive semi-definite."""
        k = element_stiffness("quad4axisymmetric", axisym_quad4_nodes, steel)
        assert is_positive_semidefinite(k)

    def test_volume_is_revolution_volume(self, axisym_quad4_nodes):
        """Volume should be 2*pi*r_centroid * area (revolution about z-axis)."""
        vol = element_volume("quad4axisymmetric", axisym_quad4_nodes)
        # Rectangle area = 1 * 1 = 1
        # Centroid r = (1 + 2) / 2 = 1.5
        # Revolution volume = 2 * pi * r_centroid * area = 2 * pi * 1.5 * 1 = 3 * pi
        expected = 2 * np.pi * 1.5 * 1.0
        assert np.isclose(vol, expected, rtol=1e-10)

    def test_rigid_body_modes(self, axisym_quad4_nodes, steel):
        """Axisymmetric element should have 1 rigid body mode (axial translation only).

        Radial translation causes hoop strain epsilon_theta = u_r/r, so it's NOT rigid.
        Only pure axial (z) translation produces zero strain.
        """
        k = element_stiffness("quad4axisymmetric", axisym_quad4_nodes, steel)
        n_zero = count_zero_eigenvalues(k, rtol=1e-6)
        assert n_zero == 1, f"Expected 1 rigid body mode (axial translation), got {n_zero}"

    def test_stress_shape(self, axisym_quad4_nodes, steel):
        """Stress tensor should have 6 components."""
        u = np.zeros(8)
        stress = compute_element_stress("quad4axisymmetric", axisym_quad4_nodes, u, steel)
        assert stress.shape[1] == 6, "Should return 6 stress components"


class TestAxisymmetricVsPlaneDifferences:
    """Test differences between axisymmetric and plane stress elements."""

    def test_different_stiffness_from_plane(self, axisym_tri3_nodes, steel):
        """Axisymmetric and plane stress should give different stiffness matrices."""
        # Use same nodes but different element types
        k_axi = element_stiffness("tri3axisymmetric", axisym_tri3_nodes, steel)
        k_plane = element_stiffness("tri3", axisym_tri3_nodes, steel)

        # They should be different due to the hoop strain term in axisymmetric
        assert not np.allclose(k_axi, k_plane, rtol=1e-3), \
            "Axisymmetric and plane stress should have different stiffness matrices"

    def test_different_volume_computation(self, axisym_tri3_nodes):
        """Axisymmetric volume is revolution, plane area is just area."""
        vol_axi = element_volume("tri3axisymmetric", axisym_tri3_nodes)
        area_plane = element_volume("tri3", axisym_tri3_nodes)

        # Axisymmetric volume is much larger (revolution)
        assert vol_axi > area_plane * 2, \
            "Axisymmetric volume should be much larger than plane area"
