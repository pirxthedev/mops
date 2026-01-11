"""Integration tests for pressure loads on faces.

Tests the Pressure load class applied to FaceQuery selections,
verifying correct conversion to consistent nodal forces.
"""

import numpy as np
import pytest

from mops import (
    Elements,
    Faces,
    Force,
    Material,
    Mesh,
    Model,
    Nodes,
    Pressure,
    solve,
)


class TestPressureLoadBasic:
    """Basic pressure load tests."""

    def test_pressure_on_hex8_top_face(self):
        """Test pressure on top face of a unit cube hex8 element.

        Pressure on top face should produce downward displacement.
        """
        # Create unit cube mesh (1x1x1)
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "hex8")

        steel = Material.steel()

        # Apply pressure on top face (z=1)
        # Pressure of 1e6 Pa (1 MPa) on 1m^2 face = 1e6 N total force
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Faces.where(z=1), Pressure(1e6))
        )

        results = solve(model)

        # Top nodes should move down (negative z)
        disp = results.displacement()
        top_nodes = Nodes.where(z=1).evaluate(mesh)
        for idx in top_nodes:
            assert disp[idx, 2] < 0, f"Node {idx} should move in -z under pressure"

        # Bottom nodes should be fixed
        bottom_nodes = Nodes.where(z=0).evaluate(mesh)
        for idx in bottom_nodes:
            assert np.allclose(disp[idx], 0.0, atol=1e-10)

    def test_pressure_equivalent_to_force(self):
        """Verify pressure load produces same result as equivalent nodal forces.

        For a 1x1 face with 4 nodes, pressure p creates total force p*A.
        With equal distribution, each node gets p*A/4.
        """
        # Create unit cube
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "hex8")

        steel = Material.steel()
        pressure = 1e6  # Pa
        area = 1.0  # m^2 (unit cube top face)
        total_force = pressure * area  # 1e6 N
        force_per_node = total_force / 4  # 4 nodes on quad face

        # Model with pressure load
        model_pressure = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Faces.where(z=1), Pressure(pressure))
        )

        # Model with equivalent nodal forces
        # Pressure pushes into surface = negative normal direction = -z
        model_force = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.where(z=1), Force(fz=-force_per_node))
        )

        results_pressure = solve(model_pressure)
        results_force = solve(model_force)

        # Displacements should match
        disp_pressure = results_pressure.displacement()
        disp_force = results_force.displacement()

        np.testing.assert_allclose(
            disp_pressure, disp_force,
            rtol=1e-10, atol=1e-15,
            err_msg="Pressure and equivalent force loads should produce same displacement"
        )

    def test_pressure_on_tet4_face(self):
        """Test pressure on a triangular face of tet4 element."""
        # Single tet4 with base on z=0 plane
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 1.0, 0.0],  # 2
            [0.5, 0.5, 1.0],  # 3 - apex
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        steel = Material.steel()

        # Apply pressure on base face (face 0: nodes 0,2,1)
        # Note: base face normal points in -z direction
        # Constrain apex fully and one base node in xy to prevent rigid body motion
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.by_indices([3]), dofs=["ux", "uy", "uz"])  # Fix apex
            .constrain(Nodes.by_indices([0]), dofs=["ux", "uy"])  # Prevent xy translation
            .constrain(Nodes.by_indices([1]), dofs=["uy"])  # Prevent rotation about z
            .load(Faces.where(z=0), Pressure(1e6))
        )

        results = solve(model)
        disp = results.displacement()

        # Apex is fixed
        assert np.allclose(disp[3], 0.0, atol=1e-10)

        # Base nodes should move (pressure pushes into surface = +z direction
        # since normal points -z)
        base_disp_z = disp[:3, 2]  # z-displacement of base nodes
        assert np.all(base_disp_z > 0), "Base nodes should move in +z under pressure"


class TestPressureLoadMultiFace:
    """Tests for pressure loads on multiple faces."""

    def test_pressure_on_boundary_faces(self):
        """Test applying pressure to all boundary faces."""
        # Create unit cube
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "hex8")

        steel = Material.steel()

        # Fix center point with minimal constraints to prevent rigid body motion
        # Actually, for uniform pressure on a cube, we need to prevent rigid body
        # motion differently. Let's fix bottom face.
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Faces.on_boundary(), Pressure(1e6))
        )

        # This should solve without error
        results = solve(model)
        assert results.max_displacement() > 0

    def test_pressure_on_two_opposite_faces(self):
        """Test pressure on opposite faces creates compression."""
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "hex8")

        steel = Material.steel()

        # Pressure on top (z=1) and bottom (z=0) faces
        # Fix middle xy plane in z
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            # Constrain z displacement at z=0 (reference plane)
            .constrain(Nodes.where(z=0), dofs=["uz"])
            # Prevent rigid body motion in xy
            .constrain(Nodes.by_indices([0]), dofs=["ux", "uy"])
            .constrain(Nodes.by_indices([1]), dofs=["uy"])
            # Pressure on top face (pushes down)
            .load(Faces.where(z=1), Pressure(1e6))
        )

        results = solve(model)

        # Top face moves down
        disp = results.displacement()
        top_nodes = Nodes.where(z=1).evaluate(mesh)
        for idx in top_nodes:
            assert disp[idx, 2] < 0


class TestPressureLoadValidation:
    """Tests for pressure load validation and error handling."""

    def test_pressure_on_nodes_raises_error(self):
        """Applying Pressure to NodeQuery should raise ValueError."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        steel = Material.steel()

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Pressure(1e6))  # Wrong! Pressure needs faces
        )

        with pytest.raises(ValueError, match="Pressure loads must be applied to faces"):
            solve(model)

    def test_zero_pressure_raises_error(self):
        """Zero pressure should raise error."""
        with pytest.raises(ValueError, match="Pressure cannot be zero"):
            Pressure(0.0)


class TestFaceAreaCalculation:
    """Tests for face area calculation used in pressure loads."""

    def test_triangular_face_area(self):
        """Test area calculation for triangular faces."""
        # Tet4 with base at z=0
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        # Face 0 is (0, 2, 1) - base triangle
        # Area = 0.5 * |v1 x v2| where v1 = (1,0,0), v2 = (0,1,0)
        # = 0.5 * |(0, 0, 1)| = 0.5
        area = mesh.get_face_area(0, 0)
        assert np.isclose(area, 0.5, rtol=1e-10)

    def test_quad_face_area(self):
        """Test area calculation for quadrilateral faces."""
        # Unit cube hex8
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "hex8")

        # Top face (face 1) is (4, 5, 6, 7) - unit square
        area = mesh.get_face_area(0, 1)
        assert np.isclose(area, 1.0, rtol=1e-10)


class TestForceOnFaces:
    """Tests for Force load on FaceQuery (distributed force)."""

    def test_force_on_face_distributed(self):
        """Force on a face should distribute to face nodes."""
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "hex8")

        steel = Material.steel()
        total_force = 4000.0  # N

        # Force on top face - should distribute to 4 nodes
        model_face = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Faces.where(z=1), Force(fz=-total_force))
        )

        # Equivalent: 1000 N per node on top face
        model_nodes = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.where(z=1), Force(fz=-total_force / 4))
        )

        results_face = solve(model_face)
        results_nodes = solve(model_nodes)

        np.testing.assert_allclose(
            results_face.displacement(),
            results_nodes.displacement(),
            rtol=1e-10, atol=1e-15
        )


class TestPressureLoadScaling:
    """Test that pressure loads scale correctly."""

    def test_double_pressure_double_displacement(self):
        """Doubling pressure should double displacement (linear elasticity)."""
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "hex8")

        steel = Material.steel()

        def create_model(pressure):
            return (
                Model(mesh, materials={"steel": steel})
                .assign(Elements.all(), material="steel")
                .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
                .load(Faces.where(z=1), Pressure(pressure))
            )

        results_1 = solve(create_model(1e6))
        results_2 = solve(create_model(2e6))

        disp_1 = results_1.displacement()
        disp_2 = results_2.displacement()

        # Displacements should scale linearly
        np.testing.assert_allclose(
            disp_2, 2.0 * disp_1,
            rtol=1e-10, atol=1e-15,
            err_msg="Displacement should scale linearly with pressure"
        )
