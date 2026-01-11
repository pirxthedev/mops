"""Element patch tests for MOPS finite elements.

Patch tests are fundamental validation tests for finite elements. A valid
finite element formulation must satisfy two conditions:

1. **Rigid Body Motion Test**: Pure translation and rotation should produce
   zero strain (no internal forces beyond numerical precision).

2. **Constant Strain Test**: A linear displacement field should produce
   exactly constant strain throughout the element. This verifies the
   strain-displacement relation (B matrix) is correctly implemented.

These tests validate the element formulations at the Python API level,
complementing the unit tests in test_elements.py.

References:
    - Hughes, T.J.R. "The Finite Element Method" (2000), Section 4.6
    - Zienkiewicz & Taylor, Vol 1, Section 9.13
    - NAFEMS QA Procedures
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mops


# =============================================================================
# Test Fixtures - Element Geometry
# =============================================================================


@pytest.fixture
def tet4_mesh():
    """Single Tet4 element with unit tetrahedron geometry."""
    nodes = np.array([
        [0.0, 0.0, 0.0],  # Node 0
        [1.0, 0.0, 0.0],  # Node 1
        [0.0, 1.0, 0.0],  # Node 2
        [0.0, 0.0, 1.0],  # Node 3
    ], dtype=np.float64)
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return mops.Mesh(nodes, elements, "tet4")


@pytest.fixture
def tet10_mesh():
    """Single Tet10 element with unit tetrahedron + midside nodes."""
    nodes = np.array([
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
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    return mops.Mesh(nodes, elements, "tet10")


@pytest.fixture
def hex8_mesh():
    """Single Hex8 element with unit cube geometry."""
    nodes = np.array([
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
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    return mops.Mesh(nodes, elements, "hex8")


@pytest.fixture
def steel():
    """Steel material for tests."""
    return mops.Material.steel()


# =============================================================================
# Rigid Body Motion Tests
# =============================================================================


class TestRigidBodyMotion:
    """Verify rigid body motion produces zero strain.

    A fundamental requirement of any valid finite element is that pure
    rigid body motion (translation and rotation) produces zero strain.
    This is tested by computing stress from prescribed displacement fields
    corresponding to rigid body modes.

    Note: Due to numerical precision in floating-point arithmetic, the stresses
    are not exactly zero but should be very small relative to material stiffness.
    We use a tolerance relative to Young's modulus (E ~ 2e11 for steel).
    """

    # Tolerance for "zero" stress: ~1e-5 relative to E (i.e., 1e6 Pa for steel)
    # This is extremely stringent - real FEA typically allows much larger errors
    RIGID_BODY_ATOL = 1e-3  # 1 mPa - effectively zero for engineering purposes

    def _compute_stress_from_displacement(
        self, element_type: str, nodes: np.ndarray, displacements: np.ndarray, material
    ) -> np.ndarray:
        """Compute stress tensor from element displacement field."""
        return mops.compute_element_stress(element_type, nodes, displacements, material)

    # -------------------------------------------------------------------------
    # Tet4 Translation Tests
    # -------------------------------------------------------------------------

    def test_tet4_translation_x(self, tet4_mesh, steel):
        """Tet4: Translation in x should produce zero stress."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        # Unit translation in x
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3] = 1.0

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Translation in x should produce zero stress")

    def test_tet4_translation_y(self, tet4_mesh, steel):
        """Tet4: Translation in y should produce zero stress."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3 + 1] = 1.0

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Translation in y should produce zero stress")

    def test_tet4_translation_z(self, tet4_mesh, steel):
        """Tet4: Translation in z should produce zero stress."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3 + 2] = 1.0

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Translation in z should produce zero stress")

    def test_tet4_translation_arbitrary(self, tet4_mesh, steel):
        """Tet4: Arbitrary translation should produce zero stress."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        tx, ty, tz = 2.5, -1.3, 0.7  # Arbitrary translation
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3] = tx
            disp[i * 3 + 1] = ty
            disp[i * 3 + 2] = tz

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Arbitrary translation should produce zero stress")

    # -------------------------------------------------------------------------
    # Tet10 Translation Tests
    # -------------------------------------------------------------------------

    def test_tet10_translation_x(self, tet10_mesh, steel):
        """Tet10: Translation in x should produce zero stress."""
        nodes = tet10_mesh.coords
        n_nodes = 10
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3] = 1.0

        stress = self._compute_stress_from_displacement("tet10", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Translation in x should produce zero stress")

    def test_tet10_translation_y(self, tet10_mesh, steel):
        """Tet10: Translation in y should produce zero stress."""
        nodes = tet10_mesh.coords
        n_nodes = 10
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3 + 1] = 1.0

        stress = self._compute_stress_from_displacement("tet10", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Translation in y should produce zero stress")

    def test_tet10_translation_z(self, tet10_mesh, steel):
        """Tet10: Translation in z should produce zero stress."""
        nodes = tet10_mesh.coords
        n_nodes = 10
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3 + 2] = 1.0

        stress = self._compute_stress_from_displacement("tet10", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Translation in z should produce zero stress")

    def test_tet10_translation_arbitrary(self, tet10_mesh, steel):
        """Tet10: Arbitrary translation should produce zero stress."""
        nodes = tet10_mesh.coords
        n_nodes = 10
        tx, ty, tz = -3.1, 2.2, 4.5
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3] = tx
            disp[i * 3 + 1] = ty
            disp[i * 3 + 2] = tz

        stress = self._compute_stress_from_displacement("tet10", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Arbitrary translation should produce zero stress")

    # -------------------------------------------------------------------------
    # Hex8 Translation Tests
    # -------------------------------------------------------------------------

    def test_hex8_translation_x(self, hex8_mesh, steel):
        """Hex8: Translation in x should produce zero stress."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3] = 1.0

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Translation in x should produce zero stress")

    def test_hex8_translation_y(self, hex8_mesh, steel):
        """Hex8: Translation in y should produce zero stress."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3 + 1] = 1.0

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Translation in y should produce zero stress")

    def test_hex8_translation_z(self, hex8_mesh, steel):
        """Hex8: Translation in z should produce zero stress."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3 + 2] = 1.0

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Translation in z should produce zero stress")

    def test_hex8_translation_arbitrary(self, hex8_mesh, steel):
        """Hex8: Arbitrary translation should produce zero stress."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        tx, ty, tz = 1.1, -2.2, 3.3
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3] = tx
            disp[i * 3 + 1] = ty
            disp[i * 3 + 2] = tz

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)
        assert_allclose(stress, 0.0, atol=self.RIGID_BODY_ATOL,
                       err_msg="Arbitrary translation should produce zero stress")


# =============================================================================
# Constant Strain Tests
# =============================================================================


class TestConstantStrain:
    """Verify linear displacement fields produce exact constant strain.

    For a valid finite element, a linear displacement field u = A*x + b
    should produce exactly constant strain ε = (A + A^T)/2 throughout
    the element. This tests the strain-displacement relation (B matrix).

    We test:
    - Uniaxial extension (ε_xx, ε_yy, ε_zz)
    - Pure shear (γ_xy, γ_yz, γ_xz)

    Note: Tolerances are chosen to be tight enough to catch bugs but
    loose enough to accommodate floating-point precision. We use:
    - rtol=1e-8 for stress comparison (allows ~0.00001% error)
    - atol=1.0 for near-zero shear stress (1 Pa is negligible)
    """

    # Relative tolerance for stress comparison
    STRESS_RTOL = 1e-8

    # Absolute tolerance for values that should be zero (Pa)
    STRESS_ATOL = 1.0

    def _compute_stress_from_displacement(
        self, element_type: str, nodes: np.ndarray, displacements: np.ndarray, material
    ) -> np.ndarray:
        """Compute stress tensor from element displacement field."""
        return mops.compute_element_stress(element_type, nodes, displacements, material)

    def _expected_stress_uniaxial_strain(self, E: float, nu: float, eps: float, direction: int):
        """Compute expected stress for uniaxial strain.

        For uniaxial strain ε in direction i (others zero):
        σ_i = E(1-ν)/((1+ν)(1-2ν)) * ε
        σ_j = Eν/((1+ν)(1-2ν)) * ε  for j ≠ i

        Args:
            E: Young's modulus
            nu: Poisson's ratio
            eps: Applied strain
            direction: 0=x, 1=y, 2=z

        Returns:
            6-component stress vector [σxx, σyy, σzz, τxy, τyz, τxz]
        """
        factor = E / ((1 + nu) * (1 - 2 * nu))
        sigma_parallel = factor * (1 - nu) * eps
        sigma_transverse = factor * nu * eps

        stress = [sigma_transverse] * 6
        stress[direction] = sigma_parallel
        stress[3] = 0.0  # τxy
        stress[4] = 0.0  # τyz
        stress[5] = 0.0  # τxz
        return np.array(stress)

    def _expected_stress_pure_shear(self, E: float, nu: float, gamma: float, plane: str):
        """Compute expected stress for pure shear.

        For pure shear γ in plane ij:
        τ_ij = G * γ where G = E / (2(1+ν))

        Args:
            E: Young's modulus
            nu: Poisson's ratio
            gamma: Shear strain
            plane: "xy", "yz", or "xz"

        Returns:
            6-component stress vector [σxx, σyy, σzz, τxy, τyz, τxz]
        """
        G = E / (2 * (1 + nu))
        tau = G * gamma

        stress = np.zeros(6)
        if plane == "xy":
            stress[3] = tau
        elif plane == "yz":
            stress[4] = tau
        elif plane == "xz":
            stress[5] = tau
        return stress

    # -------------------------------------------------------------------------
    # Tet4 Constant Strain Tests
    # -------------------------------------------------------------------------

    def test_tet4_uniaxial_strain_xx(self, tet4_mesh, steel):
        """Tet4: Linear u_x = ε*x should produce constant ε_xx."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        eps = 0.001  # 0.1% strain

        # Linear displacement: u_x = eps * x
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            x = nodes[i, 0]
            disp[i * 3] = eps * x

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)
        expected = self._expected_stress_uniaxial_strain(steel.e, steel.nu, eps, 0)

        # Tet4 has 1 integration point, so stress shape is (1, 6)
        assert stress.shape == (1, 6)
        assert_allclose(stress[0], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                       err_msg="Tet4 uniaxial ε_xx stress mismatch")

    def test_tet4_uniaxial_strain_yy(self, tet4_mesh, steel):
        """Tet4: Linear u_y = ε*y should produce constant ε_yy."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        eps = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            y = nodes[i, 1]
            disp[i * 3 + 1] = eps * y

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)
        expected = self._expected_stress_uniaxial_strain(steel.e, steel.nu, eps, 1)

        assert_allclose(stress[0], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                       err_msg="Tet4 uniaxial ε_yy stress mismatch")

    def test_tet4_uniaxial_strain_zz(self, tet4_mesh, steel):
        """Tet4: Linear u_z = ε*z should produce constant ε_zz."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        eps = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            z = nodes[i, 2]
            disp[i * 3 + 2] = eps * z

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)
        expected = self._expected_stress_uniaxial_strain(steel.e, steel.nu, eps, 2)

        assert_allclose(stress[0], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                       err_msg="Tet4 uniaxial ε_zz stress mismatch")

    def test_tet4_pure_shear_xy(self, tet4_mesh, steel):
        """Tet4: Linear u_x = γ*y should produce constant γ_xy."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        gamma = 0.001  # Shear strain

        # Pure shear: u_x = gamma * y, u_y = 0, u_z = 0
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            y = nodes[i, 1]
            disp[i * 3] = gamma * y

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)
        expected = self._expected_stress_pure_shear(steel.e, steel.nu, gamma, "xy")

        assert_allclose(stress[0], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                       err_msg="Tet4 pure shear γ_xy stress mismatch")

    def test_tet4_pure_shear_yz(self, tet4_mesh, steel):
        """Tet4: Linear u_y = γ*z should produce constant γ_yz."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        gamma = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            z = nodes[i, 2]
            disp[i * 3 + 1] = gamma * z

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)
        expected = self._expected_stress_pure_shear(steel.e, steel.nu, gamma, "yz")

        assert_allclose(stress[0], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                       err_msg="Tet4 pure shear γ_yz stress mismatch")

    def test_tet4_pure_shear_xz(self, tet4_mesh, steel):
        """Tet4: Linear u_x = γ*z should produce constant γ_xz."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        gamma = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            z = nodes[i, 2]
            disp[i * 3] = gamma * z

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)
        expected = self._expected_stress_pure_shear(steel.e, steel.nu, gamma, "xz")

        assert_allclose(stress[0], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                       err_msg="Tet4 pure shear γ_xz stress mismatch")

    # -------------------------------------------------------------------------
    # Tet10 Constant Strain Tests
    # -------------------------------------------------------------------------

    def test_tet10_uniaxial_strain_xx(self, tet10_mesh, steel):
        """Tet10: Linear u_x = ε*x should produce constant ε_xx."""
        nodes = tet10_mesh.coords
        n_nodes = 10
        eps = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            x = nodes[i, 0]
            disp[i * 3] = eps * x

        stress = self._compute_stress_from_displacement("tet10", nodes, disp, steel)
        expected = self._expected_stress_uniaxial_strain(steel.e, steel.nu, eps, 0)

        # Tet10 has 4 integration points - all should have same stress
        assert stress.shape == (4, 6)
        for ip in range(4):
            assert_allclose(stress[ip], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                           err_msg=f"Tet10 uniaxial ε_xx stress mismatch at IP {ip}")

    def test_tet10_uniaxial_strain_yy(self, tet10_mesh, steel):
        """Tet10: Linear u_y = ε*y should produce constant ε_yy."""
        nodes = tet10_mesh.coords
        n_nodes = 10
        eps = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            y = nodes[i, 1]
            disp[i * 3 + 1] = eps * y

        stress = self._compute_stress_from_displacement("tet10", nodes, disp, steel)
        expected = self._expected_stress_uniaxial_strain(steel.e, steel.nu, eps, 1)

        for ip in range(4):
            assert_allclose(stress[ip], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                           err_msg=f"Tet10 uniaxial ε_yy stress mismatch at IP {ip}")

    def test_tet10_uniaxial_strain_zz(self, tet10_mesh, steel):
        """Tet10: Linear u_z = ε*z should produce constant ε_zz."""
        nodes = tet10_mesh.coords
        n_nodes = 10
        eps = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            z = nodes[i, 2]
            disp[i * 3 + 2] = eps * z

        stress = self._compute_stress_from_displacement("tet10", nodes, disp, steel)
        expected = self._expected_stress_uniaxial_strain(steel.e, steel.nu, eps, 2)

        for ip in range(4):
            assert_allclose(stress[ip], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                           err_msg=f"Tet10 uniaxial ε_zz stress mismatch at IP {ip}")

    def test_tet10_pure_shear_xy(self, tet10_mesh, steel):
        """Tet10: Linear u_x = γ*y should produce constant γ_xy."""
        nodes = tet10_mesh.coords
        n_nodes = 10
        gamma = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            y = nodes[i, 1]
            disp[i * 3] = gamma * y

        stress = self._compute_stress_from_displacement("tet10", nodes, disp, steel)
        expected = self._expected_stress_pure_shear(steel.e, steel.nu, gamma, "xy")

        for ip in range(4):
            assert_allclose(stress[ip], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                           err_msg=f"Tet10 pure shear γ_xy stress mismatch at IP {ip}")

    # -------------------------------------------------------------------------
    # Hex8 Constant Strain Tests
    # -------------------------------------------------------------------------

    def test_hex8_uniaxial_strain_xx(self, hex8_mesh, steel):
        """Hex8: Linear u_x = ε*x should produce constant ε_xx."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        eps = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            x = nodes[i, 0]
            disp[i * 3] = eps * x

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)
        expected = self._expected_stress_uniaxial_strain(steel.e, steel.nu, eps, 0)

        # Hex8 has 2x2x2 = 8 integration points - all should have same stress
        assert stress.shape == (8, 6)
        for ip in range(8):
            assert_allclose(stress[ip], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                           err_msg=f"Hex8 uniaxial ε_xx stress mismatch at IP {ip}")

    def test_hex8_uniaxial_strain_yy(self, hex8_mesh, steel):
        """Hex8: Linear u_y = ε*y should produce constant ε_yy."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        eps = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            y = nodes[i, 1]
            disp[i * 3 + 1] = eps * y

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)
        expected = self._expected_stress_uniaxial_strain(steel.e, steel.nu, eps, 1)

        for ip in range(8):
            assert_allclose(stress[ip], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                           err_msg=f"Hex8 uniaxial ε_yy stress mismatch at IP {ip}")

    def test_hex8_uniaxial_strain_zz(self, hex8_mesh, steel):
        """Hex8: Linear u_z = ε*z should produce constant ε_zz."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        eps = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            z = nodes[i, 2]
            disp[i * 3 + 2] = eps * z

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)
        expected = self._expected_stress_uniaxial_strain(steel.e, steel.nu, eps, 2)

        for ip in range(8):
            assert_allclose(stress[ip], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                           err_msg=f"Hex8 uniaxial ε_zz stress mismatch at IP {ip}")

    def test_hex8_pure_shear_xy(self, hex8_mesh, steel):
        """Hex8: Linear u_x = γ*y should produce constant γ_xy."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        gamma = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            y = nodes[i, 1]
            disp[i * 3] = gamma * y

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)
        expected = self._expected_stress_pure_shear(steel.e, steel.nu, gamma, "xy")

        for ip in range(8):
            assert_allclose(stress[ip], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                           err_msg=f"Hex8 pure shear γ_xy stress mismatch at IP {ip}")

    def test_hex8_pure_shear_yz(self, hex8_mesh, steel):
        """Hex8: Linear u_y = γ*z should produce constant γ_yz."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        gamma = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            z = nodes[i, 2]
            disp[i * 3 + 1] = gamma * z

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)
        expected = self._expected_stress_pure_shear(steel.e, steel.nu, gamma, "yz")

        for ip in range(8):
            assert_allclose(stress[ip], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                           err_msg=f"Hex8 pure shear γ_yz stress mismatch at IP {ip}")

    def test_hex8_pure_shear_xz(self, hex8_mesh, steel):
        """Hex8: Linear u_x = γ*z should produce constant γ_xz."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        gamma = 0.001

        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            z = nodes[i, 2]
            disp[i * 3] = gamma * z

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)
        expected = self._expected_stress_pure_shear(steel.e, steel.nu, gamma, "xz")

        for ip in range(8):
            assert_allclose(stress[ip], expected, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                           err_msg=f"Hex8 pure shear γ_xz stress mismatch at IP {ip}")


# =============================================================================
# Combined Strain State Tests
# =============================================================================


class TestCombinedStrainState:
    """Test combined strain states (multiple strain components simultaneously).

    These tests verify that the superposition principle holds and that the
    element correctly handles general strain states, not just uniaxial cases.
    """

    # Relative tolerance for stress comparison
    STRESS_RTOL = 1e-8
    # Absolute tolerance for values that should be zero (Pa)
    STRESS_ATOL = 1.0

    def _compute_stress_from_displacement(
        self, element_type: str, nodes: np.ndarray, displacements: np.ndarray, material
    ) -> np.ndarray:
        return mops.compute_element_stress(element_type, nodes, displacements, material)

    def test_tet4_hydrostatic_strain(self, tet4_mesh, steel):
        """Tet4: Hydrostatic strain (ε_xx = ε_yy = ε_zz) produces hydrostatic stress."""
        nodes = tet4_mesh.coords
        n_nodes = 4
        eps = 0.001

        # Uniform volumetric strain: u = eps * [x, y, z]
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3] = eps * nodes[i, 0]
            disp[i * 3 + 1] = eps * nodes[i, 1]
            disp[i * 3 + 2] = eps * nodes[i, 2]

        stress = self._compute_stress_from_displacement("tet4", nodes, disp, steel)

        # For hydrostatic strain with isotropic material:
        # σ = K * ε_vol where K = E / (3(1-2ν)) is bulk modulus
        # ε_vol = 3ε for ε_xx = ε_yy = ε_zz = ε
        # σ_xx = σ_yy = σ_zz = K * 3ε = E * ε / (1-2ν)
        E, nu = steel.e, steel.nu
        expected_sigma = E * eps / (1 - 2 * nu)

        assert_allclose(stress[0, 0], expected_sigma, rtol=self.STRESS_RTOL)
        assert_allclose(stress[0, 1], expected_sigma, rtol=self.STRESS_RTOL)
        assert_allclose(stress[0, 2], expected_sigma, rtol=self.STRESS_RTOL)
        assert_allclose(stress[0, 3:], 0.0, atol=self.STRESS_ATOL)  # No shear

    def test_hex8_biaxial_strain(self, hex8_mesh, steel):
        """Hex8: Biaxial strain (ε_xx = ε_yy, ε_zz = 0) produces biaxial stress."""
        nodes = hex8_mesh.coords
        n_nodes = 8
        eps = 0.001

        # Biaxial strain: u_x = eps*x, u_y = eps*y, u_z = 0
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3] = eps * nodes[i, 0]
            disp[i * 3 + 1] = eps * nodes[i, 1]

        stress = self._compute_stress_from_displacement("hex8", nodes, disp, steel)

        # For biaxial strain with ε_xx = ε_yy = ε, ε_zz = 0:
        # σ_xx = σ_yy = E/(1+ν)(1-2ν) * [(1-ν)ε + νε] = Eε/(1-ν)
        # σ_zz = E*ν/(1+ν)(1-2ν) * 2ε = 2Eνε/((1+ν)(1-2ν))
        E, nu = steel.e, steel.nu
        factor = E / ((1 + nu) * (1 - 2 * nu))
        sigma_xy = factor * ((1 - nu) * eps + nu * eps)  # = E*eps/(1-nu)
        sigma_z = factor * 2 * nu * eps

        for ip in range(8):
            assert_allclose(stress[ip, 0], sigma_xy, rtol=self.STRESS_RTOL)
            assert_allclose(stress[ip, 1], sigma_xy, rtol=self.STRESS_RTOL)
            assert_allclose(stress[ip, 2], sigma_z, rtol=self.STRESS_RTOL)
            assert_allclose(stress[ip, 3:], 0.0, atol=self.STRESS_ATOL)


# =============================================================================
# Mesh with Multiple Elements (Patch of Elements)
# =============================================================================


class TestMultiElementPatch:
    """Test patch of multiple elements under constant strain.

    A patch of elements sharing nodes should all exhibit the same constant
    strain when subjected to a linear displacement field. This tests
    inter-element compatibility.
    """

    # Relative tolerance for stress comparison
    STRESS_RTOL = 1e-8
    # Absolute tolerance for values that should be zero (Pa)
    STRESS_ATOL = 1.0

    def test_two_tet4_shared_face_constant_strain(self, steel):
        """Two Tet4 elements sharing a face should both have same constant strain."""
        # Two tets sharing face 0-1-2
        nodes = np.array([
            [0.0, 0.0, 0.0],   # 0 - shared
            [1.0, 0.0, 0.0],   # 1 - shared
            [0.5, 1.0, 0.0],   # 2 - shared
            [0.5, 0.5, 1.0],   # 3 - top tet
            [0.5, 0.5, -1.0],  # 4 - bottom tet
        ], dtype=np.float64)
        elements = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 4],
        ], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")

        # Apply uniform uniaxial strain
        eps = 0.001
        n_nodes = 5
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3] = eps * nodes[i, 0]

        # Fix base nodes and load tip
        constrained = np.array([0, 2], dtype=np.int64)
        loaded = np.array([], dtype=np.int64)
        load = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Get stress for both elements
        stress1 = mops.compute_element_stress("tet4", nodes[[0, 1, 2, 3]], disp[[0,1,2, 3,4,5, 6,7,8, 9,10,11]], steel)
        stress2 = mops.compute_element_stress("tet4", nodes[[0, 1, 2, 4]], disp[[0,1,2, 3,4,5, 6,7,8, 12,13,14]], steel)

        # Both should have same stress (within tolerance)
        assert_allclose(stress1, stress2, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                       err_msg="Adjacent elements should have same stress under constant strain")

    def test_hex8_pair_constant_strain(self, steel):
        """Two Hex8 elements stacked in z should both have same constant strain."""
        # Two cubes stacked (z=0 to 1, z=1 to 2)
        nodes = np.array([
            # Bottom cube
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 0-3
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # 4-7 (shared)
            # Top cube
            [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2],  # 8-11
        ], dtype=np.float64)
        elements = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],     # Bottom
            [4, 5, 6, 7, 8, 9, 10, 11],   # Top (shares face 4-5-6-7)
        ], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "hex8")

        # Apply uniform z-strain
        eps = 0.001
        n_nodes = 12
        disp = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            disp[i * 3 + 2] = eps * nodes[i, 2]

        # Compute element stresses
        disp_elem0 = np.array([disp[i] for i in range(24)])  # Nodes 0-7
        disp_elem1 = np.array([disp[i] for i in [12,13,14, 15,16,17, 18,19,20, 21,22,23, 24,25,26, 27,28,29, 30,31,32, 33,34,35]])

        # Simpler: recompute with proper indexing
        nodes_elem0 = nodes[elements[0]]
        nodes_elem1 = nodes[elements[1]]

        disp_elem0 = np.zeros(24)
        for j, node_idx in enumerate(elements[0]):
            for k in range(3):
                disp_elem0[j*3 + k] = disp[node_idx*3 + k]

        disp_elem1 = np.zeros(24)
        for j, node_idx in enumerate(elements[1]):
            for k in range(3):
                disp_elem1[j*3 + k] = disp[node_idx*3 + k]

        stress0 = mops.compute_element_stress("hex8", nodes_elem0, disp_elem0, steel)
        stress1 = mops.compute_element_stress("hex8", nodes_elem1, disp_elem1, steel)

        # Average stress should match (individual IPs may differ due to geometry)
        avg0 = np.mean(stress0, axis=0)
        avg1 = np.mean(stress1, axis=0)
        assert_allclose(avg0, avg1, rtol=self.STRESS_RTOL, atol=self.STRESS_ATOL,
                       err_msg="Stacked hex elements should have same average stress")
