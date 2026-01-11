"""Pytest configuration and fixtures for MOPS tests.

This module provides reusable fixtures for:
- Materials (steel, aluminum, custom)
- Meshes (tet4, tet10, hex8 - single element and simple structures)
- Common test configurations

Usage:
    def test_something(steel, single_tet4_mesh):
        results = solve_simple(single_tet4_mesh, steel, ...)
"""

import numpy as np
import pytest

from mops import Material, Mesh, SolverConfig


# =============================================================================
# Material Fixtures
# =============================================================================


@pytest.fixture
def steel() -> Material:
    """Steel material (E=200 GPa, nu=0.3, rho=7850 kg/m^3)."""
    return Material.steel()


@pytest.fixture
def aluminum() -> Material:
    """Aluminum material (E=68.9 GPa, nu=0.33, rho=2700 kg/m^3)."""
    return Material.aluminum()


@pytest.fixture
def soft_material() -> Material:
    """Soft material for testing large deformations (E=1 MPa, nu=0.3)."""
    return Material("soft", e=1e6, nu=0.3)


@pytest.fixture
def nearly_incompressible() -> Material:
    """Nearly incompressible material (nu=0.499) for testing volumetric locking."""
    return Material("rubber-like", e=1e7, nu=0.499)


# =============================================================================
# Node/Element Arrays for Tet4
# =============================================================================


@pytest.fixture
def single_tet4_nodes() -> np.ndarray:
    """Node coordinates for a single unit tetrahedron."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def single_tet4_elements() -> np.ndarray:
    """Element connectivity for a single tetrahedron."""
    return np.array([[0, 1, 2, 3]], dtype=np.int64)


@pytest.fixture
def single_tet4_mesh(single_tet4_nodes, single_tet4_elements) -> Mesh:
    """Single tet4 element mesh."""
    return Mesh.from_arrays(single_tet4_nodes, single_tet4_elements, "tet4")


@pytest.fixture
def two_tet4_nodes() -> np.ndarray:
    """Node coordinates for a two-tetrahedra mesh."""
    return np.array([
        [0.0, 0.0, 0.0],  # 0 - shared
        [1.0, 0.0, 0.0],  # 1 - shared
        [0.0, 1.0, 0.0],  # 2 - shared
        [0.0, 0.0, 1.0],  # 3 - tet 1 apex
        [1.0, 1.0, 1.0],  # 4 - tet 2 apex
    ], dtype=np.float64)


@pytest.fixture
def two_tet4_elements() -> np.ndarray:
    """Element connectivity for two tetrahedra sharing a face."""
    return np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 4],
    ], dtype=np.int64)


@pytest.fixture
def two_tet4_mesh(two_tet4_nodes, two_tet4_elements) -> Mesh:
    """Two tet4 elements sharing a face."""
    return Mesh.from_arrays(two_tet4_nodes, two_tet4_elements, "tet4")


# =============================================================================
# Node/Element Arrays for Tet10
# =============================================================================


@pytest.fixture
def single_tet10_nodes() -> np.ndarray:
    """Node coordinates for a single 10-node tetrahedron.

    Node numbering (corner nodes 0-3, mid-edge nodes 4-9):
    - Corners: 0, 1, 2, 3
    - Edge 0-1: node 4
    - Edge 1-2: node 5
    - Edge 0-2: node 6
    - Edge 0-3: node 7
    - Edge 1-3: node 8
    - Edge 2-3: node 9
    """
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
def single_tet10_elements() -> np.ndarray:
    """Element connectivity for a single 10-node tetrahedron."""
    return np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)


@pytest.fixture
def single_tet10_mesh(single_tet10_nodes, single_tet10_elements) -> Mesh:
    """Single tet10 element mesh."""
    return Mesh.from_arrays(single_tet10_nodes, single_tet10_elements, "tet10")


# =============================================================================
# Node/Element Arrays for Hex8
# =============================================================================


@pytest.fixture
def unit_cube_hex8_nodes() -> np.ndarray:
    """Node coordinates for a unit cube (hex8).

    Node numbering follows standard FEA convention:
    - Bottom face (z=0): 0, 1, 2, 3 (counter-clockwise from origin)
    - Top face (z=1): 4, 5, 6, 7 (counter-clockwise)
    """
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
def unit_cube_hex8_elements() -> np.ndarray:
    """Element connectivity for a unit cube hex8."""
    return np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)


@pytest.fixture
def unit_cube_hex8_mesh(unit_cube_hex8_nodes, unit_cube_hex8_elements) -> Mesh:
    """Single hex8 unit cube mesh."""
    return Mesh.from_arrays(unit_cube_hex8_nodes, unit_cube_hex8_elements, "hex8")


@pytest.fixture
def two_hex8_nodes() -> np.ndarray:
    """Node coordinates for two hex8 elements stacked in z-direction."""
    return np.array([
        # Bottom element
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        # Shared face (z=1)
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
        # Top element top face (z=2)
        [0.0, 0.0, 2.0],  # 8
        [1.0, 0.0, 2.0],  # 9
        [1.0, 1.0, 2.0],  # 10
        [0.0, 1.0, 2.0],  # 11
    ], dtype=np.float64)


@pytest.fixture
def two_hex8_elements() -> np.ndarray:
    """Element connectivity for two stacked hex8 elements."""
    return np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],      # Bottom element
        [4, 5, 6, 7, 8, 9, 10, 11],    # Top element
    ], dtype=np.int64)


@pytest.fixture
def two_hex8_mesh(two_hex8_nodes, two_hex8_elements) -> Mesh:
    """Two hex8 elements stacked in z-direction."""
    return Mesh.from_arrays(two_hex8_nodes, two_hex8_elements, "hex8")


# =============================================================================
# Cantilever Beam Fixtures
# =============================================================================


@pytest.fixture
def cantilever_tet4_mesh() -> Mesh:
    """Simple cantilever beam with tet4 elements for basic testing.

    The beam extends in the x-direction with nodes suitable for
    constraining the x=0 face and loading the free end.

    Returns:
        Mesh: A tet4 mesh with 5 nodes and 2 elements.
    """
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
    return Mesh.from_arrays(nodes, elements, "tet4")


@pytest.fixture
def cantilever_hex8_mesh() -> Mesh:
    """Cantilever beam with hex8 elements.

    A 2x1x1 beam (two elements in x-direction) suitable for
    constraining the x=0 face and loading the x=2 face.

    Returns:
        Mesh: A hex8 mesh with 12 nodes and 2 elements.
    """
    nodes = np.array([
        # x=0 face (fixed)
        [0.0, 0.0, 0.0],  # 0
        [0.0, 1.0, 0.0],  # 1
        [0.0, 1.0, 1.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        # x=1 face (middle)
        [1.0, 0.0, 0.0],  # 4
        [1.0, 1.0, 0.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [1.0, 0.0, 1.0],  # 7
        # x=2 face (free end)
        [2.0, 0.0, 0.0],  # 8
        [2.0, 1.0, 0.0],  # 9
        [2.0, 1.0, 1.0],  # 10
        [2.0, 0.0, 1.0],  # 11
    ], dtype=np.float64)
    elements = np.array([
        [0, 4, 5, 1, 3, 7, 6, 2],   # Element 1 (x=0 to x=1)
        [4, 8, 9, 5, 7, 11, 10, 6], # Element 2 (x=1 to x=2)
    ], dtype=np.int64)
    return Mesh.from_arrays(nodes, elements, "hex8")


# =============================================================================
# Boundary Condition Fixtures
# =============================================================================


@pytest.fixture
def fix_first_four_nodes() -> np.ndarray:
    """Constrained node indices for fixing nodes 0-3."""
    return np.array([0, 1, 2, 3], dtype=np.int64)


@pytest.fixture
def load_last_node() -> np.ndarray:
    """Loaded node index for the last node (index 4)."""
    return np.array([4], dtype=np.int64)


@pytest.fixture
def downward_force() -> np.ndarray:
    """1000 N force in negative z-direction."""
    return np.array([0.0, 0.0, -1000.0], dtype=np.float64)


@pytest.fixture
def x_direction_force() -> np.ndarray:
    """1000 N force in positive x-direction."""
    return np.array([1000.0, 0.0, 0.0], dtype=np.float64)


# =============================================================================
# Solver Configuration Fixtures
# =============================================================================


@pytest.fixture
def default_solver_config() -> SolverConfig:
    """Default solver configuration."""
    return SolverConfig()


@pytest.fixture
def direct_solver_config() -> SolverConfig:
    """Direct solver configuration."""
    return SolverConfig(solver_type="direct")


# =============================================================================
# Utility Functions
# =============================================================================


def assert_displacement_at_fixed_nodes_zero(
    displacement: np.ndarray,
    fixed_nodes: np.ndarray,
    atol: float = 1e-10,
) -> None:
    """Assert that displacements at fixed nodes are zero.

    Args:
        displacement: Displacement array of shape (n_nodes, 3).
        fixed_nodes: Array of fixed node indices.
        atol: Absolute tolerance for comparison.
    """
    for node_idx in fixed_nodes:
        assert np.allclose(displacement[node_idx], 0.0, atol=atol), (
            f"Node {node_idx} should have zero displacement, got {displacement[node_idx]}"
        )


def assert_positive_displacement(
    displacement: np.ndarray,
    node_idx: int,
    direction: int,
) -> None:
    """Assert that a node has positive displacement in a given direction.

    Args:
        displacement: Displacement array of shape (n_nodes, 3).
        node_idx: Index of the node to check.
        direction: Direction index (0=x, 1=y, 2=z).
    """
    assert displacement[node_idx, direction] > 0, (
        f"Node {node_idx} should have positive displacement in direction {direction}, "
        f"got {displacement[node_idx, direction]}"
    )


def assert_negative_displacement(
    displacement: np.ndarray,
    node_idx: int,
    direction: int,
) -> None:
    """Assert that a node has negative displacement in a given direction.

    Args:
        displacement: Displacement array of shape (n_nodes, 3).
        node_idx: Index of the node to check.
        direction: Direction index (0=x, 1=y, 2=z).
    """
    assert displacement[node_idx, direction] < 0, (
        f"Node {node_idx} should have negative displacement in direction {direction}, "
        f"got {displacement[node_idx, direction]}"
    )
