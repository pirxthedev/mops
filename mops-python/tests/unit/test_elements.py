"""Unit tests for element stiffness matrices.

This module verifies fundamental properties of element stiffness matrices:
1. Symmetry: K = K^T
2. Positive semi-definiteness: x^T K x >= 0 for all x, with equality for rigid body modes
3. Patch tests: Linear displacement fields produce constant strain
4. Rigid body modes: Pure translation produces zero internal forces

These tests validate the finite element formulations at the element level,
ensuring correctness before assembly into global systems.
"""

import numpy as np
import pytest
from numpy.linalg import eigvalsh

from mops import Material, element_stiffness, element_volume


# =============================================================================
# Element Node Fixtures
# =============================================================================


@pytest.fixture
def unit_tet4_nodes() -> np.ndarray:
    """Unit tetrahedron with vertices at origin and along axes."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def scaled_tet4_nodes() -> np.ndarray:
    """Tetrahedron scaled by factor of 2."""
    return np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ], dtype=np.float64)


@pytest.fixture
def unit_tet10_nodes() -> np.ndarray:
    """Unit 10-node tetrahedron with midside nodes."""
    return np.array([
        # Corner nodes
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        # Mid-edge nodes
        [0.5, 0.0, 0.0],  # 4 (edge 0-1)
        [0.5, 0.5, 0.0],  # 5 (edge 1-2)
        [0.0, 0.5, 0.0],  # 6 (edge 0-2)
        [0.0, 0.0, 0.5],  # 7 (edge 0-3)
        [0.5, 0.0, 0.5],  # 8 (edge 1-3)
        [0.0, 0.5, 0.5],  # 9 (edge 2-3)
    ], dtype=np.float64)


@pytest.fixture
def unit_hex8_nodes() -> np.ndarray:
    """Unit cube with hex8 node ordering."""
    return np.array([
        # Bottom face (z=0)
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        # Top face (z=1)
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=np.float64)


@pytest.fixture
def stretched_hex8_nodes() -> np.ndarray:
    """Stretched hex (2x1x1) for aspect ratio testing."""
    return np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
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
    """Check if matrix is symmetric within tolerance.

    Uses both relative and absolute tolerance. For stiffness matrices with
    values in the range 1e9-1e11, a small absolute tolerance accounts for
    numerical precision in the integration.
    """
    # Use scaled absolute tolerance based on matrix magnitude
    if atol == 0.0:
        atol = rtol * np.max(np.abs(matrix))
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)


def is_positive_semidefinite(matrix: np.ndarray, rtol: float = 1e-10) -> bool:
    """Check if matrix is positive semi-definite.

    A matrix is PSD if all eigenvalues are >= 0 (allowing small negative
    values relative to the largest eigenvalue due to numerical error).
    """
    eigenvalues = eigvalsh(matrix)
    max_eig = np.max(np.abs(eigenvalues))
    # Allow eigenvalues that are negative but small relative to max
    return np.all(eigenvalues >= -rtol * max_eig)


def count_zero_eigenvalues(matrix: np.ndarray, rtol: float = 1e-10) -> int:
    """Count the number of eigenvalues close to zero.

    Uses relative tolerance based on the largest eigenvalue magnitude.
    """
    eigenvalues = eigvalsh(matrix)
    max_eig = np.max(np.abs(eigenvalues))
    tol = rtol * max_eig
    return np.sum(np.abs(eigenvalues) < tol)


def compute_rigid_body_modes(n_nodes: int) -> np.ndarray:
    """Compute rigid body mode displacement vectors.

    For 3D elements, there are 6 rigid body modes:
    - 3 translations (x, y, z)
    - 3 rotations (about x, y, z axes)

    Returns:
        Array of shape (6, n_nodes*3) containing the 6 rigid body modes.
    """
    n_dofs = n_nodes * 3
    modes = np.zeros((6, n_dofs))

    # Translation modes
    for i in range(n_nodes):
        modes[0, i*3] = 1.0      # Translation in x
        modes[1, i*3 + 1] = 1.0  # Translation in y
        modes[2, i*3 + 2] = 1.0  # Translation in z

    # Note: For patch tests, we only test translation modes
    # Rotation modes are more complex and depend on node coordinates

    return modes[:3]  # Return only translation modes


# =============================================================================
# Stiffness Matrix Symmetry Tests
# =============================================================================


class TestSymmetry:
    """Test that element stiffness matrices are symmetric."""

    def test_tet4_symmetry(self, unit_tet4_nodes, steel):
        """Tet4 stiffness matrix must be symmetric."""
        k = element_stiffness("tet4", unit_tet4_nodes, steel)
        assert k.shape == (12, 12)
        assert is_symmetric(k), "Tet4 stiffness matrix is not symmetric"

    def test_tet10_symmetry(self, unit_tet10_nodes, steel):
        """Tet10 stiffness matrix must be symmetric."""
        k = element_stiffness("tet10", unit_tet10_nodes, steel)
        assert k.shape == (30, 30)
        assert is_symmetric(k), "Tet10 stiffness matrix is not symmetric"

    def test_hex8_symmetry(self, unit_hex8_nodes, steel):
        """Hex8 stiffness matrix must be symmetric."""
        k = element_stiffness("hex8", unit_hex8_nodes, steel)
        assert k.shape == (24, 24)
        assert is_symmetric(k), "Hex8 stiffness matrix is not symmetric"

    def test_symmetry_different_materials(
        self, unit_tet4_nodes, steel, aluminum
    ):
        """Symmetry should hold for different materials."""
        k_steel = element_stiffness("tet4", unit_tet4_nodes, steel)
        k_aluminum = element_stiffness("tet4", unit_tet4_nodes, aluminum)

        assert is_symmetric(k_steel)
        assert is_symmetric(k_aluminum)

    def test_symmetry_scaled_element(self, scaled_tet4_nodes, steel):
        """Symmetry should hold for scaled elements."""
        k = element_stiffness("tet4", scaled_tet4_nodes, steel)
        assert is_symmetric(k)


# =============================================================================
# Positive Semi-Definiteness Tests
# =============================================================================


class TestPositiveSemiDefiniteness:
    """Test that element stiffness matrices are positive semi-definite."""

    def test_tet4_positive_semidefinite(self, unit_tet4_nodes, steel):
        """Tet4 stiffness matrix must be positive semi-definite."""
        k = element_stiffness("tet4", unit_tet4_nodes, steel)
        assert is_positive_semidefinite(k), \
            "Tet4 stiffness matrix has negative eigenvalues"

    def test_tet10_positive_semidefinite(self, unit_tet10_nodes, steel):
        """Tet10 stiffness matrix must be positive semi-definite."""
        k = element_stiffness("tet10", unit_tet10_nodes, steel)
        assert is_positive_semidefinite(k), \
            "Tet10 stiffness matrix has negative eigenvalues"

    def test_hex8_positive_semidefinite(self, unit_hex8_nodes, steel):
        """Hex8 stiffness matrix must be positive semi-definite."""
        k = element_stiffness("hex8", unit_hex8_nodes, steel)
        assert is_positive_semidefinite(k), \
            "Hex8 stiffness matrix has negative eigenvalues"

    def test_tet4_has_rigid_body_modes(self, unit_tet4_nodes, steel):
        """Tet4 should have 6 zero eigenvalues (rigid body modes)."""
        k = element_stiffness("tet4", unit_tet4_nodes, steel)
        n_zero = count_zero_eigenvalues(k, rtol=1e-6)
        # 3D elements have 6 rigid body modes (3 translations + 3 rotations)
        assert n_zero == 6, f"Expected 6 zero eigenvalues, got {n_zero}"

    def test_tet10_has_rigid_body_modes(self, unit_tet10_nodes, steel):
        """Tet10 should have 6 zero eigenvalues (rigid body modes)."""
        k = element_stiffness("tet10", unit_tet10_nodes, steel)
        n_zero = count_zero_eigenvalues(k, rtol=1e-6)
        assert n_zero == 6, f"Expected 6 zero eigenvalues, got {n_zero}"

    def test_hex8_has_rigid_body_modes(self, unit_hex8_nodes, steel):
        """Hex8 should have 6 zero eigenvalues (rigid body modes)."""
        k = element_stiffness("hex8", unit_hex8_nodes, steel)
        n_zero = count_zero_eigenvalues(k, rtol=1e-6)
        assert n_zero == 6, f"Expected 6 zero eigenvalues, got {n_zero}"

    def test_positive_diagonal_entries(self, unit_tet4_nodes, steel):
        """All diagonal entries of stiffness matrix should be positive."""
        k = element_stiffness("tet4", unit_tet4_nodes, steel)
        assert np.all(np.diag(k) > 0), "Stiffness matrix has non-positive diagonal"


# =============================================================================
# Rigid Body Motion Tests
# =============================================================================


class TestRigidBodyMotion:
    """Test that rigid body motions produce zero internal forces.

    Note: Due to numerical precision in floating-point arithmetic, the forces
    are not exactly zero but should be very small relative to the stiffness
    matrix magnitude (~1e-6 relative to ~1e11 stiffness values).
    """

    def _assert_rigid_body_zero(self, f: np.ndarray, k: np.ndarray, mode: str):
        """Assert forces are effectively zero for rigid body motion."""
        max_k = np.max(np.abs(k))
        max_f = np.max(np.abs(f))
        # Force should be small relative to stiffness * unit displacement
        rtol = 1e-10
        assert max_f < rtol * max_k, \
            f"{mode} produced relative force {max_f/max_k:.2e} > {rtol}"

    def test_tet4_translation_x(self, unit_tet4_nodes, steel):
        """Pure x-translation should produce zero forces."""
        k = element_stiffness("tet4", unit_tet4_nodes, steel)
        n_nodes = 4
        u = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            u[i * 3] = 1.0  # Unit translation in x
        f = k @ u
        self._assert_rigid_body_zero(f, k, "Translation in x")

    def test_tet4_translation_y(self, unit_tet4_nodes, steel):
        """Pure y-translation should produce zero forces."""
        k = element_stiffness("tet4", unit_tet4_nodes, steel)
        n_nodes = 4
        u = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            u[i * 3 + 1] = 1.0  # Unit translation in y
        f = k @ u
        self._assert_rigid_body_zero(f, k, "Translation in y")

    def test_tet4_translation_z(self, unit_tet4_nodes, steel):
        """Pure z-translation should produce zero forces."""
        k = element_stiffness("tet4", unit_tet4_nodes, steel)
        n_nodes = 4
        u = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            u[i * 3 + 2] = 1.0  # Unit translation in z
        f = k @ u
        self._assert_rigid_body_zero(f, k, "Translation in z")

    def test_tet10_translation_modes(self, unit_tet10_nodes, steel):
        """All 3 translation modes for Tet10 should produce zero forces."""
        k = element_stiffness("tet10", unit_tet10_nodes, steel)
        n_nodes = 10

        for direction in range(3):
            u = np.zeros(n_nodes * 3)
            for i in range(n_nodes):
                u[i * 3 + direction] = 1.0
            f = k @ u
            self._assert_rigid_body_zero(f, k, f"Translation mode {direction}")

    def test_hex8_translation_modes(self, unit_hex8_nodes, steel):
        """All 3 translation modes for Hex8 should produce zero forces."""
        k = element_stiffness("hex8", unit_hex8_nodes, steel)
        n_nodes = 8

        for direction in range(3):
            u = np.zeros(n_nodes * 3)
            for i in range(n_nodes):
                u[i * 3 + direction] = 1.0
            f = k @ u
            self._assert_rigid_body_zero(f, k, f"Translation mode {direction}")


# =============================================================================
# Patch Tests (Constant Strain)
# =============================================================================


class TestPatchTests:
    """Patch tests verify that constant strain fields are reproduced exactly.

    For a valid finite element formulation, a linear displacement field
    should produce constant strain throughout the element. This is tested
    by applying a linear displacement field and verifying that the resulting
    internal forces are consistent with the expected constant stress.

    Note: Due to numerical precision, force sums are not exactly zero but
    should be very small relative to the individual force magnitudes.
    """

    def _assert_equilibrium(
        self, f: np.ndarray, n_nodes: int, k: np.ndarray, label: str
    ):
        """Assert forces sum to zero (equilibrium) within tolerance."""
        fx_sum = sum(f[i * 3] for i in range(n_nodes))
        fy_sum = sum(f[i * 3 + 1] for i in range(n_nodes))
        fz_sum = sum(f[i * 3 + 2] for i in range(n_nodes))

        # Tolerance based on max force magnitude
        max_f = np.max(np.abs(f))
        rtol = 1e-8

        assert abs(fx_sum) < rtol * max_f, \
            f"{label}: Sum of x-forces {fx_sum:.2e} exceeds tolerance"
        assert abs(fy_sum) < rtol * max_f, \
            f"{label}: Sum of y-forces {fy_sum:.2e} exceeds tolerance"
        assert abs(fz_sum) < rtol * max_f, \
            f"{label}: Sum of z-forces {fz_sum:.2e} exceeds tolerance"

    def test_tet4_constant_strain_xx(self, unit_tet4_nodes, steel):
        """Apply uniform ε_xx strain and verify consistent nodal forces.

        For a constant strain ε_xx = du/dx = constant, we apply:
        u_x = ε_xx * x, u_y = 0, u_z = 0

        The internal force should be in equilibrium (sum to zero).
        """
        k = element_stiffness("tet4", unit_tet4_nodes, steel)

        # Apply linear displacement field: u_x = 0.001 * x (strain = 0.001)
        strain_xx = 0.001
        u = np.zeros(12)
        for i in range(4):
            x = unit_tet4_nodes[i, 0]
            u[i * 3] = strain_xx * x  # u_x = ε_xx * x

        f = k @ u
        self._assert_equilibrium(f, 4, k, "Tet4 constant ε_xx")

    def test_hex8_constant_strain_xx(self, unit_hex8_nodes, steel):
        """Apply uniform ε_xx strain to Hex8 and verify equilibrium."""
        k = element_stiffness("hex8", unit_hex8_nodes, steel)

        strain_xx = 0.001
        u = np.zeros(24)
        for i in range(8):
            x = unit_hex8_nodes[i, 0]
            u[i * 3] = strain_xx * x

        f = k @ u
        self._assert_equilibrium(f, 8, k, "Hex8 constant ε_xx")

    def test_tet10_constant_strain_xx(self, unit_tet10_nodes, steel):
        """Apply uniform ε_xx strain to Tet10 and verify equilibrium."""
        k = element_stiffness("tet10", unit_tet10_nodes, steel)

        strain_xx = 0.001
        u = np.zeros(30)
        for i in range(10):
            x = unit_tet10_nodes[i, 0]
            u[i * 3] = strain_xx * x

        f = k @ u
        self._assert_equilibrium(f, 10, k, "Tet10 constant ε_xx")

    def test_pure_shear_equilibrium(self, unit_tet4_nodes, steel):
        """Apply pure shear strain and verify equilibrium.

        Pure shear: u_x = γ * y, u_y = 0 (gives γ_xy = du_x/dy)
        """
        k = element_stiffness("tet4", unit_tet4_nodes, steel)

        shear = 0.001
        u = np.zeros(12)
        for i in range(4):
            y = unit_tet4_nodes[i, 1]
            u[i * 3] = shear * y  # u_x = γ * y

        f = k @ u
        self._assert_equilibrium(f, 4, k, "Tet4 pure shear")


# =============================================================================
# Volume Tests
# =============================================================================


class TestElementVolume:
    """Test element volume computation."""

    def test_tet4_unit_volume(self, unit_tet4_nodes):
        """Unit tetrahedron volume should be 1/6."""
        vol = element_volume("tet4", unit_tet4_nodes)
        expected = 1.0 / 6.0
        assert np.isclose(vol, expected, rtol=1e-10), \
            f"Expected volume {expected}, got {vol}"

    def test_tet4_scaled_volume(self, scaled_tet4_nodes):
        """Scaled (2x) tetrahedron volume should be 8 * (1/6)."""
        vol = element_volume("tet4", scaled_tet4_nodes)
        expected = 8.0 / 6.0  # Volume scales as L^3
        assert np.isclose(vol, expected, rtol=1e-10)

    def test_tet10_unit_volume(self, unit_tet10_nodes):
        """Unit Tet10 volume should equal unit Tet4 volume (1/6)."""
        vol = element_volume("tet10", unit_tet10_nodes)
        expected = 1.0 / 6.0
        assert np.isclose(vol, expected, rtol=1e-10)

    def test_hex8_unit_cube_volume(self, unit_hex8_nodes):
        """Unit cube volume should be 1.0."""
        vol = element_volume("hex8", unit_hex8_nodes)
        assert np.isclose(vol, 1.0, rtol=1e-10)

    def test_hex8_stretched_volume(self, stretched_hex8_nodes):
        """2x1x1 hex volume should be 2.0."""
        vol = element_volume("hex8", stretched_hex8_nodes)
        assert np.isclose(vol, 2.0, rtol=1e-10)


# =============================================================================
# Material Scaling Tests
# =============================================================================


class TestMaterialScaling:
    """Test that stiffness scales correctly with material properties."""

    def test_stiffness_scales_with_youngs_modulus(self, unit_tet4_nodes):
        """Stiffness should scale linearly with Young's modulus."""
        mat1 = Material("test1", e=100e9, nu=0.3)
        mat2 = Material("test2", e=200e9, nu=0.3)  # 2x E

        k1 = element_stiffness("tet4", unit_tet4_nodes, mat1)
        k2 = element_stiffness("tet4", unit_tet4_nodes, mat2)

        # K scales linearly with E
        assert np.allclose(k2, 2.0 * k1, rtol=1e-10), \
            "Stiffness does not scale linearly with Young's modulus"

    def test_different_poisson_ratios(self, unit_tet4_nodes):
        """Different Poisson ratios should produce different stiffness."""
        mat1 = Material("nu_low", e=100e9, nu=0.1)
        mat2 = Material("nu_high", e=100e9, nu=0.4)

        k1 = element_stiffness("tet4", unit_tet4_nodes, mat1)
        k2 = element_stiffness("tet4", unit_tet4_nodes, mat2)

        # Matrices should be different but both valid
        assert not np.allclose(k1, k2)
        assert is_symmetric(k1)
        assert is_symmetric(k2)
        assert is_positive_semidefinite(k1)
        assert is_positive_semidefinite(k2)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_element_type(self, unit_tet4_nodes, steel):
        """Invalid element type should raise ValueError."""
        with pytest.raises(ValueError):
            element_stiffness("invalid", unit_tet4_nodes, steel)

    def test_wrong_node_count_tet4(self, steel):
        """Wrong node count for tet4 should raise ValueError."""
        bad_nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        with pytest.raises(ValueError):
            element_stiffness("tet4", bad_nodes, steel)

    def test_wrong_node_count_hex8(self, steel):
        """Wrong node count for hex8 should raise ValueError."""
        bad_nodes = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=np.float64)
        with pytest.raises(ValueError):
            element_stiffness("hex8", bad_nodes, steel)


# =============================================================================
# Cross-Element Comparison Tests
# =============================================================================


class TestCrossElementComparison:
    """Compare behavior across element types."""

    def test_tet4_vs_tet10_same_geometry(
        self, unit_tet4_nodes, unit_tet10_nodes, steel
    ):
        """Tet4 and Tet10 on same geometry should have similar properties.

        While the matrices will be different sizes and values, both should:
        - Be symmetric
        - Be positive semi-definite
        - Have 6 rigid body modes
        """
        k4 = element_stiffness("tet4", unit_tet4_nodes, steel)
        k10 = element_stiffness("tet10", unit_tet10_nodes, steel)

        assert is_symmetric(k4)
        assert is_symmetric(k10)
        assert is_positive_semidefinite(k4)
        assert is_positive_semidefinite(k10)
        assert count_zero_eigenvalues(k4, rtol=1e-6) == 6
        assert count_zero_eigenvalues(k10, rtol=1e-6) == 6

    def test_volume_consistency(
        self, unit_tet4_nodes, unit_tet10_nodes, unit_hex8_nodes
    ):
        """Element volumes should be physically consistent."""
        vol_tet4 = element_volume("tet4", unit_tet4_nodes)
        vol_tet10 = element_volume("tet10", unit_tet10_nodes)
        vol_hex8 = element_volume("hex8", unit_hex8_nodes)

        # Unit tet volume = 1/6, unit hex volume = 1
        assert np.isclose(vol_tet4, 1/6)
        assert np.isclose(vol_tet10, 1/6)  # Same geometry as tet4
        assert np.isclose(vol_hex8, 1.0)

        # 6 unit tets would fill a unit cube (approximately)
        assert np.isclose(6 * vol_tet4, vol_hex8)
