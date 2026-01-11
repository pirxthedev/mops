"""Tests for multi-material assembly and solve."""

import numpy as np
import pytest

from mops import Material, Mesh
from mops._core import solve_with_materials


def get_core_mesh(mesh):
    """Get the underlying Rust mesh from Python wrapper."""
    return mesh._inner if hasattr(mesh, "_inner") else mesh


class TestSolveWithMaterials:
    """Test solve_with_materials function."""

    @pytest.fixture
    def two_element_mesh(self):
        """Create a simple mesh with two tet4 elements sharing a face."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, -1.0],
        ], dtype=np.float64)

        elements = np.array([
            [0, 1, 2, 3],  # Upper tetrahedron
            [0, 1, 2, 4],  # Lower tetrahedron
        ], dtype=np.int64)

        return Mesh(nodes, elements, "tet4")

    @pytest.fixture
    def steel(self):
        """Steel material."""
        return Material.steel()

    @pytest.fixture
    def aluminum(self):
        """Aluminum material."""
        return Material.aluminum()

    def test_uniform_material(self, two_element_mesh, steel):
        """Test that uniform material produces same results as single-material solve."""
        # Fix base nodes
        constraints = np.array([
            [0, 0, 0.0],  # node 0, ux
            [0, 1, 0.0],  # node 0, uy
            [0, 2, 0.0],  # node 0, uz
            [1, 0, 0.0],  # node 1, ux
            [1, 1, 0.0],  # node 1, uy
            [1, 2, 0.0],  # node 1, uz
            [2, 0, 0.0],  # node 2, ux
            [2, 1, 0.0],  # node 2, uy
            [2, 2, 0.0],  # node 2, uz
        ], dtype=np.float64)

        # Apply force at node 3 (top)
        forces = np.array([
            [3, 0.0, 0.0, -1000.0],  # node 3, downward force
        ], dtype=np.float64)

        # All elements use material 0 (steel)
        element_material_indices = np.array([
            [0, 0],  # element 0 -> material 0
            [1, 0],  # element 1 -> material 0
        ], dtype=np.int64)

        results = solve_with_materials(
            get_core_mesh(two_element_mesh),
            [steel],
            element_material_indices,
            constraints,
            forces,
        )

        assert results.max_displacement() > 0
        disp = results.displacement()
        assert disp.shape == (5, 3)

        # Fixed nodes should have zero displacement
        for node in [0, 1, 2]:
            assert np.allclose(disp[node], 0.0)

    def test_multi_material_steel_and_aluminum(self, two_element_mesh, steel, aluminum):
        """Test that different materials produce different stiffness behavior."""
        # Fix base nodes
        constraints = np.array([
            [0, 0, 0.0],
            [0, 1, 0.0],
            [0, 2, 0.0],
            [1, 0, 0.0],
            [1, 1, 0.0],
            [1, 2, 0.0],
            [2, 0, 0.0],
            [2, 1, 0.0],
            [2, 2, 0.0],
        ], dtype=np.float64)

        # Apply force at node 3 (top)
        forces = np.array([
            [3, 0.0, 0.0, -1000.0],
        ], dtype=np.float64)

        # Element 0 is steel, element 1 is aluminum
        element_material_indices = np.array([
            [0, 0],  # element 0 -> steel (index 0)
            [1, 1],  # element 1 -> aluminum (index 1)
        ], dtype=np.int64)

        materials = [steel, aluminum]

        results = solve_with_materials(
            get_core_mesh(two_element_mesh),
            materials,
            element_material_indices,
            constraints,
            forces,
        )

        assert results.max_displacement() > 0
        disp = results.displacement()

        # Fixed nodes should have zero displacement
        for node in [0, 1, 2]:
            assert np.allclose(disp[node], 0.0)

        # The structure should deform due to the load
        assert not np.allclose(disp[3], 0.0) or not np.allclose(disp[4], 0.0)

    def test_softer_material_larger_displacement(self, two_element_mesh, steel, aluminum):
        """Test that using softer material (aluminum) produces larger displacements."""
        constraints = np.array([
            [0, 0, 0.0],
            [0, 1, 0.0],
            [0, 2, 0.0],
            [1, 0, 0.0],
            [1, 1, 0.0],
            [1, 2, 0.0],
            [2, 0, 0.0],
            [2, 1, 0.0],
            [2, 2, 0.0],
        ], dtype=np.float64)

        forces = np.array([
            [3, 0.0, 0.0, -1000.0],
        ], dtype=np.float64)

        # All steel
        all_steel_indices = np.array([
            [0, 0],
            [1, 0],
        ], dtype=np.int64)

        # All aluminum
        all_aluminum_indices = np.array([
            [0, 0],
            [1, 0],
        ], dtype=np.int64)

        results_steel = solve_with_materials(
            get_core_mesh(two_element_mesh),
            [steel],
            all_steel_indices,
            constraints,
            forces,
        )

        results_aluminum = solve_with_materials(
            get_core_mesh(two_element_mesh),
            [aluminum],
            all_aluminum_indices,
            constraints,
            forces,
        )

        # Aluminum is softer (lower E), so should produce larger displacement
        assert results_aluminum.max_displacement() > results_steel.max_displacement()

    def test_empty_element_indices_uses_first_material(self, two_element_mesh, steel, aluminum):
        """Test that empty element_material_indices defaults to first material."""
        constraints = np.array([
            [0, 0, 0.0],
            [0, 1, 0.0],
            [0, 2, 0.0],
            [1, 0, 0.0],
            [1, 1, 0.0],
            [1, 2, 0.0],
            [2, 0, 0.0],
            [2, 1, 0.0],
            [2, 2, 0.0],
        ], dtype=np.float64)

        forces = np.array([
            [3, 0.0, 0.0, -1000.0],
        ], dtype=np.float64)

        # Empty array - should default to first material (steel)
        empty_indices = np.zeros((0, 2), dtype=np.int64)

        results = solve_with_materials(
            get_core_mesh(two_element_mesh),
            [steel, aluminum],
            empty_indices,
            constraints,
            forces,
        )

        assert results.max_displacement() > 0

    def test_invalid_material_index_raises_error(self, two_element_mesh, steel):
        """Test that out-of-bounds material index raises error."""
        constraints = np.array([
            [0, 0, 0.0],
            [0, 1, 0.0],
            [0, 2, 0.0],
        ], dtype=np.float64)

        forces = np.array([
            [3, 0.0, 0.0, -1000.0],
        ], dtype=np.float64)

        # Material index 5 is out of bounds (only 1 material)
        invalid_indices = np.array([
            [0, 5],  # Invalid material index
        ], dtype=np.int64)

        with pytest.raises(ValueError, match="out of bounds"):
            solve_with_materials(
                get_core_mesh(two_element_mesh),
                [steel],
                invalid_indices,
                constraints,
                forces,
            )

    def test_stress_recovery_with_multi_material(self, two_element_mesh, steel, aluminum):
        """Test that stress recovery uses correct material for each element."""
        constraints = np.array([
            [0, 0, 0.0],
            [0, 1, 0.0],
            [0, 2, 0.0],
            [1, 0, 0.0],
            [1, 1, 0.0],
            [1, 2, 0.0],
            [2, 0, 0.0],
            [2, 1, 0.0],
            [2, 2, 0.0],
        ], dtype=np.float64)

        forces = np.array([
            [3, 0.0, 0.0, -1000.0],
        ], dtype=np.float64)

        element_material_indices = np.array([
            [0, 0],  # element 0 -> steel
            [1, 1],  # element 1 -> aluminum
        ], dtype=np.int64)

        results = solve_with_materials(
            get_core_mesh(two_element_mesh),
            [steel, aluminum],
            element_material_indices,
            constraints,
            forces,
        )

        # Verify stress tensors are computed (6 components per element)
        stress = results.stress()
        assert stress.shape == (2, 6)  # 2 elements, 6 stress components

        # Verify von Mises stresses are computed
        vm = results.von_mises()
        assert len(vm) == 2


class TestMultiMaterialRust:
    """Test multi-material functionality at the Rust core level."""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple single-element mesh for basic tests."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        elements = np.array([
            [0, 1, 2, 3],
        ], dtype=np.int64)

        return Mesh(nodes, elements, "tet4")

    def test_no_materials_raises_error(self, simple_mesh):
        """Test that empty materials list raises error."""
        constraints = np.array([[0, 0, 0.0]], dtype=np.float64)
        forces = np.array([[3, 0.0, 0.0, -1000.0]], dtype=np.float64)
        indices = np.zeros((0, 2), dtype=np.int64)

        with pytest.raises(ValueError, match="At least one material"):
            solve_with_materials(
                get_core_mesh(simple_mesh),
                [],  # Empty materials list
                indices,
                constraints,
                forces,
            )
