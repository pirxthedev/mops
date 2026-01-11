"""Unit tests for individual DOF constraint support.

This module verifies that the solver correctly applies constraints to individual
degrees of freedom (DOFs), enabling:
1. Symmetry boundary conditions (e.g., ux=0 only at x=0 plane)
2. Sliding constraints (fix normal but allow tangential motion)
3. Mixed boundary conditions
"""

import numpy as np
import pytest

from mops import Material, Mesh, solve_simple
from tests.conftest import nodes_to_constraints


class TestIndividualDOFConstraints:
    """Test constraining individual DOFs at nodes."""

    @pytest.fixture
    def steel(self) -> Material:
        return Material.steel()

    @pytest.fixture
    def unit_cube_mesh(self) -> Mesh:
        """Single hex8 unit cube."""
        nodes = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_constrain_single_dof_ux(self, unit_cube_mesh, steel):
        """Constrain only ux at a node, uy and uz remain free."""
        # Constrain ux=0 at nodes 0,3,4,7 (x=0 face)
        constraint_rows = []
        for node in [0, 3, 4, 7]:
            constraint_rows.append([float(node), 0.0, 0.0])  # ux = 0

        # Also fix all DOFs at node 0 to prevent rigid body motion
        constraint_rows.extend([
            [0.0, 1.0, 0.0],  # uy = 0
            [0.0, 2.0, 0.0],  # uz = 0
            [3.0, 2.0, 0.0],  # uz = 0 at node 3
        ])

        constraints = np.array(constraint_rows, dtype=np.float64)

        # Apply force in x direction on x=1 face
        loaded_nodes = np.array([1, 2, 5, 6], dtype=np.int64)
        load = np.array([1000.0, 0.0, 0.0], dtype=np.float64)

        results = solve_simple(unit_cube_mesh, steel, constraints, loaded_nodes, load)
        disp = results.displacement()

        # Check ux=0 at constrained nodes
        for node in [0, 3, 4, 7]:
            assert abs(disp[node, 0]) < 1e-10, f"Node {node} ux should be 0"

        # But uy and uz at nodes 3,4,7 can be non-zero due to Poisson effect
        # (Poisson's ratio causes contraction perpendicular to tension)

        # Loaded nodes should have positive ux
        for node in [1, 2, 5, 6]:
            assert disp[node, 0] > 0, f"Node {node} should have positive ux"

    def test_constrain_single_dof_uy_symmetry(self, unit_cube_mesh, steel):
        """Symmetry BC: uy=0 at y=0 face, allowing ux and uz motion."""
        # Constrain uy=0 at nodes on y=0 face (nodes 0,1,4,5)
        constraint_rows = []
        for node in [0, 1, 4, 5]:
            constraint_rows.append([float(node), 1.0, 0.0])  # uy = 0

        # Prevent rigid body motion
        constraint_rows.extend([
            [0.0, 0.0, 0.0],  # ux = 0 at node 0
            [0.0, 2.0, 0.0],  # uz = 0 at node 0
            [1.0, 2.0, 0.0],  # uz = 0 at node 1
        ])

        constraints = np.array(constraint_rows, dtype=np.float64)

        # Apply force in y direction on y=1 face
        loaded_nodes = np.array([2, 3, 6, 7], dtype=np.int64)
        load = np.array([0.0, 500.0, 0.0], dtype=np.float64)

        results = solve_simple(unit_cube_mesh, steel, constraints, loaded_nodes, load)
        disp = results.displacement()

        # uy=0 at y=0 face nodes
        for node in [0, 1, 4, 5]:
            assert abs(disp[node, 1]) < 1e-10, f"Node {node} uy should be 0"

        # Loaded nodes should have positive uy
        for node in [2, 3, 6, 7]:
            assert disp[node, 1] > 0, f"Node {node} should have positive uy"

    def test_sliding_constraint(self, unit_cube_mesh, steel):
        """Sliding: fix uz only at bottom face, allow ux and uy."""
        # Constrain uz=0 at z=0 face (nodes 0,1,2,3)
        constraint_rows = []
        for node in [0, 1, 2, 3]:
            constraint_rows.append([float(node), 2.0, 0.0])  # uz = 0

        # Prevent rigid body translation/rotation in x-y plane
        constraint_rows.extend([
            [0.0, 0.0, 0.0],  # ux = 0 at node 0
            [0.0, 1.0, 0.0],  # uy = 0 at node 0
            [1.0, 1.0, 0.0],  # uy = 0 at node 1
        ])

        constraints = np.array(constraint_rows, dtype=np.float64)

        # Apply downward force on top face
        loaded_nodes = np.array([4, 5, 6, 7], dtype=np.int64)
        load = np.array([0.0, 0.0, -1000.0], dtype=np.float64)

        results = solve_simple(unit_cube_mesh, steel, constraints, loaded_nodes, load)
        disp = results.displacement()

        # uz=0 at z=0 face
        for node in [0, 1, 2, 3]:
            assert abs(disp[node, 2]) < 1e-10, f"Node {node} uz should be 0"

        # Top face should move down (negative uz)
        for node in [4, 5, 6, 7]:
            assert disp[node, 2] < 0, f"Node {node} should have negative uz"

        # Nodes 1,2,3 can slide in x-y plane due to Poisson expansion
        # Check that they're not all zero (would indicate over-constraint)
        slide_magnitudes = [
            np.sqrt(disp[n, 0]**2 + disp[n, 1]**2) for n in [1, 2, 3]
        ]
        assert sum(slide_magnitudes) > 0, "Nodes should slide in x-y plane"

    def test_prescribed_nonzero_displacement(self, unit_cube_mesh, steel):
        """Test prescribing a non-zero displacement value."""
        prescribed_disp = 0.001  # 1mm displacement

        # Fix bottom face completely
        constraint_rows = []
        for node in [0, 1, 2, 3]:
            for dof in [0, 1, 2]:
                constraint_rows.append([float(node), float(dof), 0.0])

        # Prescribe ux = 0.001 on top face
        for node in [4, 5, 6, 7]:
            constraint_rows.append([float(node), 0.0, prescribed_disp])  # ux = 0.001
            constraint_rows.append([float(node), 1.0, 0.0])  # uy = 0
            constraint_rows.append([float(node), 2.0, 0.0])  # uz = 0

        constraints = np.array(constraint_rows, dtype=np.float64)

        # No external load - pure displacement BC
        loaded_nodes = np.array([], dtype=np.int64)
        load = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        results = solve_simple(unit_cube_mesh, steel, constraints, loaded_nodes, load)
        disp = results.displacement()

        # Check prescribed displacement at top face
        for node in [4, 5, 6, 7]:
            assert np.isclose(disp[node, 0], prescribed_disp, atol=1e-10), \
                f"Node {node} ux should be {prescribed_disp}"

        # Bottom face should have zero displacement
        for node in [0, 1, 2, 3]:
            assert np.allclose(disp[node], 0.0, atol=1e-10)


class TestNodesToConstraints:
    """Test the nodes_to_constraints helper function."""

    def test_default_all_dofs(self):
        """With no dofs specified, all 3 DOFs should be constrained."""
        nodes = np.array([1, 5, 10])
        constraints = nodes_to_constraints(nodes)

        # Should have 9 rows (3 nodes x 3 DOFs)
        assert constraints.shape == (9, 3)

        # Check all nodes have all 3 DOFs
        for i, node in enumerate([1, 5, 10]):
            for dof in [0, 1, 2]:
                row_idx = i * 3 + dof
                assert constraints[row_idx, 0] == float(node)
                assert constraints[row_idx, 1] == float(dof)
                assert constraints[row_idx, 2] == 0.0

    def test_single_dof(self):
        """Constrain only specific DOFs."""
        nodes = np.array([0, 1])
        constraints = nodes_to_constraints(nodes, dofs=[1])  # Only uy

        # Should have 2 rows
        assert constraints.shape == (2, 3)
        assert constraints[0, 0] == 0.0  # node 0
        assert constraints[0, 1] == 1.0  # dof 1 (uy)
        assert constraints[1, 0] == 1.0  # node 1
        assert constraints[1, 1] == 1.0  # dof 1 (uy)

    def test_two_dofs(self):
        """Constrain two DOFs per node."""
        nodes = np.array([0])
        constraints = nodes_to_constraints(nodes, dofs=[0, 2])  # ux and uz

        assert constraints.shape == (2, 3)
        assert constraints[0, 1] == 0.0  # dof ux
        assert constraints[1, 1] == 2.0  # dof uz

    def test_nonzero_value(self):
        """Prescribe a non-zero displacement value."""
        nodes = np.array([3])
        constraints = nodes_to_constraints(nodes, dofs=[0], value=0.005)

        assert constraints.shape == (1, 3)
        assert constraints[0, 2] == 0.005

    def test_empty_nodes(self):
        """Empty node array produces empty constraints."""
        nodes = np.array([], dtype=np.int64)
        constraints = nodes_to_constraints(nodes)

        assert constraints.shape == (0, 3)
