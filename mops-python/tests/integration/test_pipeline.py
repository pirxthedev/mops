"""Integration tests for the full solve pipeline.

Tests the complete workflow: mesh -> Model -> solve -> Results
using the declarative Python API with copy-on-write semantics.
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


class TestModelAPIPipeline:
    """Test the full Model API pipeline with various element types."""

    def test_tet4_cantilever_model_api(self):
        """Test cantilever beam with tet4 elements using Model API."""
        # Create mesh
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0 - fixed end
            [1.0, 0.0, 0.0],  # 1 - fixed end
            [0.0, 1.0, 0.0],  # 2 - fixed end
            [0.0, 0.0, 1.0],  # 3 - fixed end
            [1.0, 1.0, 1.0],  # 4 - free end, load point
        ], dtype=np.float64)
        elements = np.array([
            [0, 1, 2, 4],
            [0, 1, 3, 4],
        ], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        # Build model with method chaining
        steel = Material.steel()
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([4]), Force(fz=-1000))
        )

        # Solve
        results = solve(model)

        # Verify results
        assert isinstance(results, Results)
        assert results.max_displacement() > 0

        disp = results.displacement()
        assert disp.shape == (5, 3)

        # Fixed nodes should have zero displacement
        fixed_indices = Nodes.where(x=0).evaluate(mesh)
        for idx in fixed_indices:
            assert np.allclose(disp[idx], 0.0, atol=1e-10)

        # Loaded node should move in negative z-direction
        assert disp[4, 2] < 0

    def test_hex8_compression_model_api(self):
        """Test unit cube under compression using Model API."""
        # Create unit cube mesh
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

        # Build model
        aluminum = Material.aluminum()
        model = (
            Model(mesh, materials={"aluminum": aluminum})
            .assign(Elements.all(), material="aluminum")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.where(z=1), Force(fz=-1000))
        )

        # Solve
        results = solve(model)

        # Verify
        assert results.max_displacement() > 0

        disp = results.displacement()

        # Bottom face fixed
        bottom_nodes = Nodes.where(z=0).evaluate(mesh)
        for idx in bottom_nodes:
            assert np.allclose(disp[idx], 0.0, atol=1e-10)

        # Top face moves down
        top_nodes = Nodes.where(z=1).evaluate(mesh)
        for idx in top_nodes:
            assert disp[idx, 2] < 0

    def test_tet10_model_api(self):
        """Test quadratic tet10 element using Model API."""
        # Create tet10 mesh (10 nodes per element)
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
        mesh = Mesh(nodes, elements, "tet10")

        # Build model constraining base triangle (nodes 0, 1, 2)
        steel = Material.steel()
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        # Solve
        results = solve(model)

        # Verify
        assert results.max_displacement() > 0
        disp = results.displacement()
        assert disp.shape == (10, 3)

        # Apex node (3) should move in negative z
        assert disp[3, 2] < 0


class TestModelImmutability:
    """Test that Model maintains copy-on-write semantics."""

    def test_model_chain_creates_new_instances(self):
        """Test that each method call creates a new Model instance."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")
        steel = Material.steel()

        # Each step creates a new model
        model1 = Model(mesh, materials={"steel": steel})
        model2 = model1.assign(Elements.all(), material="steel")
        model3 = model2.constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
        model4 = model3.load(Nodes.by_indices([3]), Force(fz=-1000))

        # All are different objects
        assert model1 is not model2
        assert model2 is not model3
        assert model3 is not model4

        # Original model still has no constraints/loads
        assert len(model1._state.constraints) == 0
        assert len(model1._state.loads) == 0
        assert len(model2._state.constraints) == 0

    def test_model_with_material_creates_copy(self):
        """Test that with_material creates a new instance."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        model1 = Model(mesh)
        model2 = model1.with_material("steel", Material.steel())

        assert model1 is not model2
        assert "steel" not in model1.materials
        assert "steel" in model2.materials


class TestQueryIntegration:
    """Test query evaluation integration with Model."""

    def test_nodes_where_x_equals(self):
        """Test selecting nodes at specific x-coordinate."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        # Select nodes at x=0
        query = Nodes.where(x=0)
        indices = query.evaluate(mesh)

        assert len(indices) == 2
        assert 0 in indices
        assert 1 in indices

    def test_nodes_where_range(self):
        """Test selecting nodes in coordinate range."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        # Select nodes with x between 0.25 and 1.25
        query = Nodes.where(x__gt=0.25, x__lt=1.25)
        indices = query.evaluate(mesh)

        assert len(indices) == 2
        assert 1 in indices  # x=0.5
        assert 2 in indices  # x=1.0

    def test_nodes_union_query(self):
        """Test union of node queries."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        # Union: x=0 OR x=3
        query = Nodes.where(x=0).union(Nodes.where(x=3))
        indices = query.evaluate(mesh)

        assert len(indices) == 2
        assert 0 in indices
        assert 3 in indices

    def test_elements_all_query(self):
        """Test selecting all elements."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 4],
        ], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        query = Elements.all()
        indices = query.evaluate(mesh)

        assert len(indices) == 2
        assert 0 in indices
        assert 1 in indices


class TestSolverConfiguration:
    """Test solver configuration in pipeline."""

    def test_solve_with_direct_solver(self):
        """Test specifying direct solver configuration."""
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
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        # Solve with explicit direct solver config
        config = SolverConfig(solver_type="direct")
        results = solve(model, config)

        assert results.max_displacement() > 0

    def test_solve_with_default_config(self):
        """Test solve with default (auto) solver configuration."""
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
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        # Solve with default config
        results = solve(model)
        assert results.max_displacement() > 0


class TestResultsExtraction:
    """Test extracting results from solved model."""

    def test_displacement_components(self):
        """Test extracting displacement field components."""
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
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        results = solve(model)

        # Check displacement field
        disp = results.displacement()
        assert disp.shape == (4, 3)
        assert disp.dtype == np.float64

        # Check displacement magnitude
        disp_mag = results.displacement_magnitude()
        assert len(disp_mag) == 4
        assert disp_mag.dtype == np.float64

        # Magnitude should match computed norm
        for i in range(4):
            expected_mag = np.linalg.norm(disp[i])
            assert np.isclose(disp_mag[i], expected_mag)

    def test_max_displacement_consistency(self):
        """Test that max_displacement matches displacement_magnitude max."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        steel = Material.steel()
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        results = solve(model)

        max_disp = results.max_displacement()
        disp_mag = results.displacement_magnitude()

        assert np.isclose(max_disp, np.max(disp_mag))


class TestErrorHandling:
    """Test error handling in the pipeline."""

    def test_solve_without_materials_raises(self):
        """Test that solving without materials raises error."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        model = Model(mesh)

        with pytest.raises(ValueError, match="No materials defined"):
            solve(model)

    def test_solve_without_constraints_raises(self):
        """Test that solving without constraints raises error."""
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
            .load(Nodes.by_indices([3]), Force(fz=-1000))
        )

        with pytest.raises(ValueError, match="No constraints defined"):
            solve(model)

    def test_solve_without_loads_raises(self):
        """Test that solving without loads raises error."""
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
        )

        with pytest.raises(ValueError, match="No loads defined"):
            solve(model)

    def test_assign_unknown_material_raises(self):
        """Test that assigning unknown material raises error."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        model = Model(mesh, materials={"steel": Material.steel()})

        with pytest.raises(ValueError, match="Unknown material"):
            model.assign(Elements.all(), material="titanium")

    def test_invalid_dof_raises(self):
        """Test that invalid DOF in constraint raises error."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        model = Model(mesh, materials={"steel": Material.steel()})

        with pytest.raises(ValueError, match="Invalid DOF"):
            model.constrain(Nodes.where(z=0), dofs=["invalid_dof"])


class TestMultiElementMeshPipeline:
    """Test pipeline with multi-element meshes."""

    def test_two_hex8_stacked(self):
        """Test two hex8 elements stacked in z-direction."""
        nodes = np.array([
            # Bottom element bottom face
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            # Shared face
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
            # Top element top face
            [0.0, 0.0, 2.0],  # 8
            [1.0, 0.0, 2.0],  # 9
            [1.0, 1.0, 2.0],  # 10
            [0.0, 1.0, 2.0],  # 11
        ], dtype=np.float64)
        elements = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],      # Bottom element
            [4, 5, 6, 7, 8, 9, 10, 11],    # Top element
        ], dtype=np.int64)
        mesh = Mesh(nodes, elements, "hex8")

        steel = Material.steel()
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.where(z=2), Force(fz=-1000))
        )

        results = solve(model)

        disp = results.displacement()
        assert disp.shape == (12, 3)

        # Bottom fixed
        bottom_nodes = Nodes.where(z=0).evaluate(mesh)
        for idx in bottom_nodes:
            assert np.allclose(disp[idx], 0.0, atol=1e-10)

        # Top moves down
        top_nodes = Nodes.where(z=2).evaluate(mesh)
        for idx in top_nodes:
            assert disp[idx, 2] < 0

        # Middle layer should move less than top
        mid_nodes = Nodes.where(z=1).evaluate(mesh)
        for mid_idx in mid_nodes:
            for top_idx in top_nodes:
                assert abs(disp[mid_idx, 2]) < abs(disp[top_idx, 2])

    def test_mixed_constraints_on_multi_element(self):
        """Test applying different constraints to multi-element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0 - fixed
            [1.0, 0.0, 0.0],  # 1 - fixed
            [0.0, 1.0, 0.0],  # 2 - fixed
            [0.0, 0.0, 1.0],  # 3 - fixed
            [2.0, 1.0, 1.0],  # 4 - loaded
        ], dtype=np.float64)
        elements = np.array([
            [0, 1, 2, 4],
            [0, 1, 3, 4],
        ], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tet4")

        steel = Material.steel()
        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.by_indices([0, 1, 2, 3]), dofs=["ux", "uy", "uz"])
            .load(Nodes.by_indices([4]), Force(fx=1000, fy=500, fz=-500))
        )

        results = solve(model)

        disp = results.displacement()

        # Verify fixed nodes
        for idx in [0, 1, 2, 3]:
            assert np.allclose(disp[idx], 0.0, atol=1e-10)

        # Loaded node should have non-zero displacement in all directions
        assert disp[4, 0] > 0  # positive x force -> positive x disp
        assert disp[4, 1] > 0  # positive y force -> positive y disp
        assert disp[4, 2] < 0  # negative z force -> negative z disp
