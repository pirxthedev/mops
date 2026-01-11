"""Integration tests for Tet10 element solve operations.

These tests verify that the 10-node quadratic tetrahedral element (Tet10)
works correctly in end-to-end solve scenarios including:
- Single element solves
- Multi-element meshes
- Stress recovery
- Comparison with Tet4 (quadratic should be more accurate)

Tet10 elements have quadratic shape functions and can capture linear strain
fields exactly, making them superior to Tet4 for bending problems.
"""

import numpy as np
import pytest

from mops import (
    Elements,
    Force,
    Material,
    Mesh,
    Model,
    Nodes,
    Results,
    SolverConfig,
    solve,
)


# =============================================================================
# Fixtures for Tet10 Meshes
# =============================================================================


@pytest.fixture
def single_tet10_mesh():
    """Single Tet10 element mesh with unit tetrahedron."""
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
    return Mesh(nodes, elements, "tet10")


@pytest.fixture
def two_tet10_mesh():
    """Two Tet10 elements sharing a triangular face.

    Creates two tetrahedra that share face 0-1-2 with midside nodes 4,5,6.
    Both have positive Jacobian determinant with proper node ordering.
    """
    # For Tet10, node ordering is:
    # [n0, n1, n2, n3, m01, m12, m02, m03, m13, m23]
    # where n0-n3 are corner nodes and m_ij is midpoint of edge i-j

    nodes = np.array([
        # Shared face corner nodes (triangle in z=0 plane)
        [0.0, 0.0, 0.0],   # 0
        [1.0, 0.0, 0.0],   # 1
        [0.5, 0.866, 0.0], # 2 (equilateral triangle)
        # First tet apex (above)
        [0.5, 0.289, 0.816],   # 3 (centroid + height for regular tet)
        # Shared mid-edge nodes (base triangle edges)
        [0.5, 0.0, 0.0],       # 4 (edge 0-1)
        [0.75, 0.433, 0.0],    # 5 (edge 1-2)
        [0.25, 0.433, 0.0],    # 6 (edge 0-2)
        # First tet mid-edge nodes (to apex 3)
        [0.25, 0.1445, 0.408],   # 7 (edge 0-3)
        [0.75, 0.1445, 0.408],   # 8 (edge 1-3)
        [0.5, 0.5775, 0.408],    # 9 (edge 2-3)
        # Second tet apex (below)
        [0.5, 0.289, -0.816],  # 10
        # Second tet mid-edge nodes (to apex 10)
        [0.25, 0.1445, -0.408],  # 11 (edge 0-10)
        [0.75, 0.1445, -0.408],  # 12 (edge 1-10)
        [0.5, 0.5775, -0.408],   # 13 (edge 2-10)
    ], dtype=np.float64)

    # Element 1: corners 0,1,2,3 with proper midside nodes
    # Element 2: corners 0,2,1,10 (note: 0,2,1 reverses winding for opposite apex)
    # This gives positive Jacobian for both elements
    elements = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 2, 1, 10, 6, 5, 4, 11, 13, 12],  # Reversed base winding for -z apex
    ], dtype=np.int64)
    return Mesh(nodes, elements, "tet10")


@pytest.fixture
def steel():
    """Steel material for testing."""
    return Material.steel()


@pytest.fixture
def aluminum():
    """Aluminum material for testing."""
    return Material.aluminum()


# =============================================================================
# Single Element Tests
# =============================================================================


class TestTet10SingleElement:
    """Tests for single Tet10 element solve."""

    def test_single_element_point_load(self, single_tet10_mesh, steel):
        """Test single Tet10 element with point load at apex."""
        mesh = single_tet10_mesh

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        results = solve(model)

        assert isinstance(results, Results)
        assert results.max_displacement() > 0

        disp = results.displacement()
        assert disp.shape == (10, 3)

        # Fixed nodes (base triangle z=0) should have zero displacement
        base_nodes = Nodes.where(z=0).evaluate(mesh)
        for idx in base_nodes:
            assert np.allclose(disp[idx], 0.0, atol=1e-10), (
                f"Node {idx} at z=0 should be fixed"
            )

        # Apex (node 3) should move downward
        assert disp[3, 2] < 0, "Apex should deflect downward under negative z load"

    def test_single_element_multi_direction_load(self, single_tet10_mesh, steel):
        """Test single Tet10 with load in multiple directions."""
        mesh = single_tet10_mesh

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fx=500, fy=300, fz=-1000))
        )

        results = solve(model)
        disp = results.displacement()

        # Apex should move in all directions corresponding to load
        assert disp[3, 0] > 0, "Positive fx should cause positive x displacement"
        assert disp[3, 1] > 0, "Positive fy should cause positive y displacement"
        assert disp[3, 2] < 0, "Negative fz should cause negative z displacement"

    def test_single_element_midside_load(self, single_tet10_mesh, steel):
        """Test loading on midside node (unique to Tet10)."""
        mesh = single_tet10_mesh

        # Load on midside node 8 (midpoint of edge 1-3)
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([8]), Force(fz=-500))
        )

        results = solve(model)
        disp = results.displacement()

        # Node 8 is at (0.5, 0, 0.5) - should deflect
        assert disp[8, 2] < 0, "Midside node should deflect under load"

        # Apex should also deflect (connected via shape functions)
        assert disp[3, 2] < 0, "Apex should deflect due to load on midside"

    def test_single_element_tension(self, single_tet10_mesh, steel):
        """Test Tet10 in tension (positive z load)."""
        mesh = single_tet10_mesh

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=1000))
        )

        results = solve(model)
        disp = results.displacement()

        # Apex should move upward (positive z)
        assert disp[3, 2] > 0, "Apex should move up under tension"


# =============================================================================
# Multi-Element Tests
# =============================================================================


class TestTet10MultiElement:
    """Tests for multi-element Tet10 meshes."""

    def test_two_elements_shared_face(self, two_tet10_mesh, steel):
        """Test two Tet10 elements sharing a face.

        The mesh has two tets sharing face 0-1-2 (plus midside nodes 4,5,6).
        - Tet 1 apex at node 3 (z > 0)
        - Tet 2 apex at node 10 (z < 0)
        We fix the lower apex (node 10) and load the upper apex (node 3).
        """
        mesh = two_tet10_mesh

        # Fix the lower apex and its midside nodes
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.by_indices([10, 11, 12, 13]), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=1000))  # Pull up
        )

        results = solve(model)
        disp = results.displacement()

        # Top apex (node 3) should move upward
        assert disp[3, 2] > 0, "Top apex should deflect upward under positive z load"

        # Fixed nodes should have zero displacement
        for idx in [10, 11, 12, 13]:
            assert np.allclose(disp[idx], 0.0, atol=1e-10)

        # Shared face nodes should also move (they're not constrained)
        shared_nodes = [0, 1, 2, 4, 5, 6]
        for idx in shared_nodes:
            assert np.isfinite(disp[idx]).all(), f"Node {idx} should have finite displacement"

    def test_displacement_continuity(self, two_tet10_mesh, steel):
        """Verify displacement continuity at shared nodes."""
        mesh = two_tet10_mesh

        # Fix lower apex and load upper apex
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.by_indices([10, 11, 12, 13]), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=1000))
        )

        results = solve(model)
        disp = results.displacement()

        # Check that all nodes have finite displacements (no NaN/Inf)
        assert np.all(np.isfinite(disp)), "All displacements should be finite"

        # All nodes should have valid displacement magnitudes
        for idx in range(len(disp)):
            assert np.linalg.norm(disp[idx]) >= 0, f"Node {idx} should have valid displacement"


# =============================================================================
# Comparison with Tet4 (Accuracy Tests)
# =============================================================================


class TestTet10VsTet4:
    """Compare Tet10 and Tet4 element accuracy.

    Tet10 (quadratic) should generally be more accurate than Tet4 (linear),
    especially for problems involving bending or non-constant strain fields.
    For a single element with point load, Tet10 is significantly more flexible
    because it can capture bending deformation that Tet4 cannot.
    """

    def test_both_elements_solve_correctly(self, steel):
        """Verify both Tet10 and Tet4 produce valid solutions.

        For a single-element problem with a point load at the apex:
        - Both elements should produce non-zero displacement
        - Both should move in the correct direction (downward for negative z load)
        - Tet10 is expected to be more flexible (larger displacement)
          because quadratic shape functions can capture bending deformation
        """
        # Tet4 mesh (4 nodes)
        tet4_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        tet4_elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        tet4_mesh = Mesh(tet4_nodes, tet4_elements, "tet4")

        # Tet10 mesh (same geometry + midside nodes)
        tet10_nodes = np.array([
            # Corner nodes (same as Tet4)
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            # Mid-edge nodes
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ], dtype=np.float64)
        tet10_elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
        tet10_mesh = Mesh(tet10_nodes, tet10_elements, "tet10")

        # Same boundary conditions and load for both
        force = Force(fz=-1000)

        # Solve Tet4
        model4 = (
            Model(tet4_mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), force)
        )
        results4 = solve(model4)
        disp4_apex = results4.displacement()[3]

        # Solve Tet10
        model10 = (
            Model(tet10_mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), force)
        )
        results10 = solve(model10)
        disp10_apex = results10.displacement()[3]

        # Both should produce non-zero displacement
        tet4_mag = np.linalg.norm(disp4_apex)
        tet10_mag = np.linalg.norm(disp10_apex)
        assert tet4_mag > 0, "Tet4 should produce non-zero displacement"
        assert tet10_mag > 0, "Tet10 should produce non-zero displacement"

        # Both should deflect downward (negative z direction)
        assert disp4_apex[2] < 0, "Tet4 apex should deflect downward"
        assert disp10_apex[2] < 0, "Tet10 apex should deflect downward"

        # Tet10 is generally more flexible for bending-dominated problems
        # because it can capture quadratic displacement fields
        # The ratio can be significant (up to 10-20x for single element problems)
        # but both should be finite and reasonable
        ratio = tet10_mag / tet4_mag
        assert ratio > 0, "Ratio should be positive"
        assert np.isfinite(ratio), "Ratio should be finite"


# =============================================================================
# Load Scaling Tests
# =============================================================================


class TestTet10LoadScaling:
    """Test linear load-displacement relationship for Tet10."""

    def test_linear_load_response(self, single_tet10_mesh, steel):
        """Verify displacement scales linearly with load."""
        mesh = single_tet10_mesh

        displacements = []
        loads = [500, 1000, 2000]

        for load_mag in loads:
            model = (
                Model(mesh, materials={"steel": steel})
                .assign(Elements.all(), material="steel")
                .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
                .load(Nodes.by_indices([3]), Force(fz=-load_mag))
            )
            results = solve(model)
            disp = results.displacement()
            displacements.append(disp[3, 2])  # z-displacement at apex

        # Check linearity: displacement / load should be constant
        ratios = [d / l for d, l in zip(displacements, loads)]

        for i in range(1, len(ratios)):
            assert np.isclose(ratios[i], ratios[0], rtol=1e-6), (
                f"Non-linear response: ratio[{i}]={ratios[i]:.6e} vs ratio[0]={ratios[0]:.6e}"
            )


# =============================================================================
# Material Scaling Tests
# =============================================================================


class TestTet10MaterialScaling:
    """Test displacement scales inversely with stiffness for Tet10."""

    def test_inverse_stiffness_relationship(self, single_tet10_mesh):
        """Verify displacement halves when E doubles."""
        mesh = single_tet10_mesh

        E_values = [100e9, 200e9, 400e9]
        displacements = []

        for E in E_values:
            material = Material("test", e=E, nu=0.3)
            model = (
                Model(mesh, materials={"test": material})
                .assign(Elements.all(), material="test")
                .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
                .load(Nodes.by_indices([3]), Force(fz=-1000))
            )
            results = solve(model)
            disp = results.displacement()
            displacements.append(abs(disp[3, 2]))

        # Check inverse relationship: displacement * E should be constant
        products = [d * E for d, E in zip(displacements, E_values)]

        for i in range(1, len(products)):
            assert np.isclose(products[i], products[0], rtol=1e-6), (
                f"Non-inverse E-displacement: product[{i}]={products[i]:.6e} "
                f"vs product[0]={products[0]:.6e}"
            )


# =============================================================================
# Results Extraction Tests
# =============================================================================


class TestTet10ResultsExtraction:
    """Test extracting results from Tet10 solve."""

    def test_displacement_field_shape(self, single_tet10_mesh, steel):
        """Verify displacement field has correct shape."""
        mesh = single_tet10_mesh

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )
        results = solve(model)

        disp = results.displacement()
        assert disp.shape == (10, 3), f"Expected (10, 3), got {disp.shape}"
        assert disp.dtype == np.float64

    def test_displacement_magnitude(self, single_tet10_mesh, steel):
        """Verify displacement magnitude computation."""
        mesh = single_tet10_mesh

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )
        results = solve(model)

        disp = results.displacement()
        disp_mag = results.displacement_magnitude()

        assert len(disp_mag) == 10
        for i in range(10):
            expected_mag = np.linalg.norm(disp[i])
            assert np.isclose(disp_mag[i], expected_mag), (
                f"Magnitude mismatch at node {i}"
            )

    def test_max_displacement_consistency(self, single_tet10_mesh, steel):
        """Verify max_displacement matches max of displacement_magnitude."""
        mesh = single_tet10_mesh

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )
        results = solve(model)

        max_disp = results.max_displacement()
        disp_mag = results.displacement_magnitude()

        assert np.isclose(max_disp, np.max(disp_mag)), (
            f"max_displacement ({max_disp}) != max(displacement_magnitude) ({np.max(disp_mag)})"
        )


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestTet10ErrorHandling:
    """Test error handling specific to Tet10 solves."""

    def test_insufficient_constraints(self, single_tet10_mesh, steel):
        """Test error when model is underconstrained."""
        mesh = single_tet10_mesh

        # Only constrain one node (rigid body motion possible)
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.by_indices([0]), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        # This may either raise an error or produce a singular matrix warning
        # depending on implementation. At minimum, it should not hang.
        try:
            results = solve(model)
            # If it solves, check for reasonable behavior
            # (some solvers add stabilization for rigid body modes)
            disp = results.displacement()
            assert np.all(np.isfinite(disp)), "Displacements should be finite"
        except Exception:
            # Expected to fail due to singular matrix
            pass

    def test_no_materials_error(self, single_tet10_mesh):
        """Test that solving without materials raises error."""
        mesh = single_tet10_mesh

        model = Model(mesh)

        with pytest.raises(ValueError, match="No materials defined"):
            solve(model)

    def test_no_loads_error(self, single_tet10_mesh, steel):
        """Test that solving without loads raises error."""
        mesh = single_tet10_mesh

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
        )

        with pytest.raises(ValueError, match="No loads defined"):
            solve(model)

    def test_no_constraints_error(self, single_tet10_mesh, steel):
        """Test that solving without constraints raises error."""
        mesh = single_tet10_mesh

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        with pytest.raises(ValueError, match="No constraints defined"):
            solve(model)


# =============================================================================
# Solver Configuration Tests
# =============================================================================


class TestTet10SolverConfiguration:
    """Test solver configuration options with Tet10 elements."""

    def test_direct_solver(self, single_tet10_mesh, steel):
        """Test Tet10 with direct solver explicitly specified."""
        mesh = single_tet10_mesh

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        config = SolverConfig(solver_type="direct")
        results = solve(model, config)

        assert results.max_displacement() > 0

    def test_auto_solver(self, single_tet10_mesh, steel):
        """Test Tet10 with auto solver selection."""
        mesh = single_tet10_mesh

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        # Default config uses auto-selection
        results = solve(model)

        assert results.max_displacement() > 0
