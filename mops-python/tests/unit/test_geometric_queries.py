"""Unit tests for geometric node selection queries.

Tests the following query methods:
- Nodes.near_point(): Select nodes within tolerance of a point
- Nodes.in_sphere(): Select nodes within a sphere
- Nodes.in_box(): Select nodes within an axis-aligned bounding box
- Nodes.near_line(): Select nodes within tolerance of a line segment
"""

import numpy as np
import pytest

from mops.query import Nodes


class MockMesh:
    """Mock mesh for testing queries without full Rust bindings."""

    def __init__(self, coords: np.ndarray):
        """Create mock mesh with given coordinates.

        Args:
            coords: Nx3 array of node coordinates
        """
        self._coords = np.asarray(coords, dtype=np.float64)

    @property
    def n_nodes(self) -> int:
        return self._coords.shape[0]

    @property
    def coords(self) -> np.ndarray:
        return self._coords


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def grid_mesh() -> MockMesh:
    """3x3x3 grid of nodes centered at origin.

    Node positions: (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), ...
    Total: 27 nodes
    """
    coords = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                coords.append([x, y, z])
    return MockMesh(np.array(coords, dtype=np.float64))


@pytest.fixture
def line_mesh() -> MockMesh:
    """Nodes along the x-axis from 0 to 10.

    Nodes at: (0,0,0), (1,0,0), (2,0,0), ..., (10,0,0)
    """
    coords = [[x, 0, 0] for x in range(11)]
    return MockMesh(np.array(coords, dtype=np.float64))


@pytest.fixture
def sparse_mesh() -> MockMesh:
    """Sparse mesh with nodes at specific known locations."""
    coords = [
        [0, 0, 0],       # 0: origin
        [10, 0, 0],      # 1: on x-axis
        [0, 10, 0],      # 2: on y-axis
        [0, 0, 10],      # 3: on z-axis
        [5, 5, 5],       # 4: diagonal
        [3, 4, 0],       # 5: distance 5 from origin in xy-plane
    ]
    return MockMesh(np.array(coords, dtype=np.float64))


# =============================================================================
# near_point tests
# =============================================================================


class TestNearPoint:
    """Test Nodes.near_point() query."""

    def test_near_point_exact_match(self, sparse_mesh):
        """Near point should find exact node location."""
        query = Nodes.near_point((0, 0, 0), tol=0.01)
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == 1
        assert 0 in indices

    def test_near_point_within_tolerance(self, sparse_mesh):
        """Near point should find nodes within tolerance."""
        # Node at (3, 4, 0) is distance 5 from origin
        query = Nodes.near_point((0, 0, 0), tol=5.5)
        indices = query.evaluate(sparse_mesh)
        # Should include origin (0) and (3,4,0) node (5)
        assert 0 in indices
        assert 5 in indices

    def test_near_point_no_match(self, sparse_mesh):
        """Near point with small tolerance may find no nodes."""
        query = Nodes.near_point((100, 100, 100), tol=0.1)
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == 0

    def test_near_point_grid_center(self, grid_mesh):
        """Near point at grid center should find center node."""
        query = Nodes.near_point((0, 0, 0), tol=0.1)
        indices = query.evaluate(grid_mesh)
        # Find the index of (0, 0, 0) in the grid
        center_idx = None
        for i, coord in enumerate(grid_mesh.coords):
            if np.allclose(coord, [0, 0, 0]):
                center_idx = i
                break
        assert center_idx in indices

    def test_near_point_multiple_nodes(self, grid_mesh):
        """Near point should find multiple nodes if within tolerance."""
        # Tolerance of 1.0 includes center + 6 face neighbors (distance 1)
        query = Nodes.near_point((0, 0, 0), tol=1.0)
        indices = query.evaluate(grid_mesh)
        # Center node plus 6 face neighbors (distance exactly 1)
        assert len(indices) == 7

    def test_near_point_repr(self):
        """Test string representation."""
        query = Nodes.near_point((1, 2, 3), tol=0.5)
        assert "near_point" in repr(query)
        assert "(1, 2, 3)" in repr(query)
        assert "0.5" in repr(query)


# =============================================================================
# in_sphere tests
# =============================================================================


class TestInSphere:
    """Test Nodes.in_sphere() query."""

    def test_in_sphere_single_node(self, sparse_mesh):
        """Small sphere should contain single node."""
        query = Nodes.in_sphere((0, 0, 0), radius=0.01)
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == 1
        assert 0 in indices

    def test_in_sphere_multiple_nodes(self, sparse_mesh):
        """Larger sphere should contain multiple nodes."""
        # Radius 5.1 from origin includes nodes at distance <= 5
        query = Nodes.in_sphere((0, 0, 0), radius=5.1)
        indices = query.evaluate(sparse_mesh)
        assert 0 in indices  # origin
        assert 5 in indices  # (3, 4, 0) at distance 5

    def test_in_sphere_all_nodes(self, sparse_mesh):
        """Very large sphere should contain all nodes."""
        query = Nodes.in_sphere((0, 0, 0), radius=100)
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == sparse_mesh.n_nodes

    def test_in_sphere_no_nodes(self, sparse_mesh):
        """Sphere far from mesh should contain no nodes."""
        query = Nodes.in_sphere((1000, 1000, 1000), radius=1)
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == 0

    def test_in_sphere_boundary_inclusive(self, sparse_mesh):
        """Node exactly on sphere boundary should be included."""
        # Node (3, 4, 0) is at distance 5 from origin
        query = Nodes.in_sphere((0, 0, 0), radius=5.0)
        indices = query.evaluate(sparse_mesh)
        assert 5 in indices

    def test_in_sphere_off_center(self, grid_mesh):
        """Sphere centered off-origin."""
        query = Nodes.in_sphere((1, 1, 1), radius=0.1)
        indices = query.evaluate(grid_mesh)
        # Find the index of (1, 1, 1) in the grid
        corner_idx = None
        for i, coord in enumerate(grid_mesh.coords):
            if np.allclose(coord, [1, 1, 1]):
                corner_idx = i
                break
        assert corner_idx in indices

    def test_in_sphere_repr(self):
        """Test string representation."""
        query = Nodes.in_sphere((1, 2, 3), radius=5.0)
        assert "in_sphere" in repr(query)
        assert "(1, 2, 3)" in repr(query)
        assert "5.0" in repr(query)


# =============================================================================
# in_box tests
# =============================================================================


class TestInBox:
    """Test Nodes.in_box() query."""

    def test_in_box_single_node(self, sparse_mesh):
        """Tight box around origin should contain single node."""
        query = Nodes.in_box((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == 1
        assert 0 in indices

    def test_in_box_all_nodes(self, sparse_mesh):
        """Box enclosing all nodes should contain all."""
        query = Nodes.in_box((-1, -1, -1), (11, 11, 11))
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == sparse_mesh.n_nodes

    def test_in_box_no_nodes(self, sparse_mesh):
        """Box far from mesh should contain no nodes."""
        query = Nodes.in_box((100, 100, 100), (200, 200, 200))
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == 0

    def test_in_box_boundary_inclusive(self, sparse_mesh):
        """Nodes on box boundary should be included."""
        # Node at (10, 0, 0) is on boundary
        query = Nodes.in_box((0, -1, -1), (10, 1, 1))
        indices = query.evaluate(sparse_mesh)
        assert 0 in indices   # origin
        assert 1 in indices   # (10, 0, 0)

    def test_in_box_partial_selection(self, grid_mesh):
        """Box should select subset of grid."""
        # Select only nodes with x >= 0
        query = Nodes.in_box((0, -2, -2), (2, 2, 2))
        indices = query.evaluate(grid_mesh)
        for i in indices:
            assert grid_mesh.coords[i, 0] >= 0

    def test_in_box_quadrant(self, grid_mesh):
        """Box selecting one quadrant."""
        query = Nodes.in_box((0, 0, 0), (2, 2, 2))
        indices = query.evaluate(grid_mesh)
        # Should include nodes at (0,0,0), (0,0,1), (0,1,0), (0,1,1),
        # (1,0,0), (1,0,1), (1,1,0), (1,1,1) = 8 nodes
        assert len(indices) == 8
        for i in indices:
            x, y, z = grid_mesh.coords[i]
            assert x >= 0 and y >= 0 and z >= 0

    def test_in_box_repr(self):
        """Test string representation."""
        query = Nodes.in_box((0, 0, 0), (10, 10, 10))
        assert "in_box" in repr(query)
        assert "(0, 0, 0)" in repr(query)
        assert "(10, 10, 10)" in repr(query)


# =============================================================================
# near_line tests
# =============================================================================


class TestNearLine:
    """Test Nodes.near_line() query."""

    def test_near_line_on_line(self, line_mesh):
        """Nodes exactly on line should be selected."""
        query = Nodes.near_line((0, 0, 0), (10, 0, 0), tol=0.1)
        indices = query.evaluate(line_mesh)
        # All 11 nodes are on the line
        assert len(indices) == 11

    def test_near_line_partial_segment(self, line_mesh):
        """Line segment should only include nodes within segment."""
        query = Nodes.near_line((2, 0, 0), (5, 0, 0), tol=0.1)
        indices = query.evaluate(line_mesh)
        # Nodes at x=2, 3, 4, 5
        assert len(indices) == 4
        for i in indices:
            x = line_mesh.coords[i, 0]
            assert 2 <= x <= 5

    def test_near_line_with_offset_nodes(self, sparse_mesh):
        """Nodes offset from line by tolerance should be included."""
        # Line along x-axis, node at (3, 4, 0) is distance 4 from line
        query = Nodes.near_line((0, 0, 0), (10, 0, 0), tol=4.1)
        indices = query.evaluate(sparse_mesh)
        assert 0 in indices   # origin (on line)
        assert 1 in indices   # (10, 0, 0) (on line)
        assert 5 in indices   # (3, 4, 0) (distance 4 from line)

    def test_near_line_no_match(self, sparse_mesh):
        """Line far from nodes should match nothing."""
        query = Nodes.near_line((100, 100, 100), (200, 200, 200), tol=1)
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == 0

    def test_near_line_degenerate_point(self, sparse_mesh):
        """Degenerate line (start == end) should act like near_point."""
        query = Nodes.near_line((0, 0, 0), (0, 0, 0), tol=0.1)
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == 1
        assert 0 in indices

    def test_near_line_perpendicular_distance(self, grid_mesh):
        """Test perpendicular distance to line."""
        # Line along z-axis through origin
        query = Nodes.near_line((0, 0, -2), (0, 0, 2), tol=0.1)
        indices = query.evaluate(grid_mesh)
        # Should only select nodes at (0, 0, z)
        for i in indices:
            x, y, z = grid_mesh.coords[i]
            assert abs(x) < 0.01 and abs(y) < 0.01

    def test_near_line_diagonal(self, sparse_mesh):
        """Test line not aligned with axes."""
        # Diagonal line from origin to (10, 10, 10)
        # Node at (5, 5, 5) is on this line
        query = Nodes.near_line((0, 0, 0), (10, 10, 10), tol=0.1)
        indices = query.evaluate(sparse_mesh)
        assert 0 in indices  # origin
        assert 4 in indices  # (5, 5, 5) on diagonal

    def test_near_line_repr(self):
        """Test string representation."""
        query = Nodes.near_line((0, 0, 0), (1, 1, 1), tol=0.5)
        assert "near_line" in repr(query)
        assert "(0, 0, 0)" in repr(query)
        assert "(1, 1, 1)" in repr(query)
        assert "0.5" in repr(query)


# =============================================================================
# Set operation integration tests
# =============================================================================


class TestGeometricQuerySetOperations:
    """Test geometric queries with set operations."""

    def test_union_of_spheres(self, sparse_mesh):
        """Union of two spheres."""
        q1 = Nodes.in_sphere((0, 0, 0), radius=0.1)  # origin only
        q2 = Nodes.in_sphere((10, 0, 0), radius=0.1)  # (10, 0, 0) only
        union = q1 | q2
        indices = union.evaluate(sparse_mesh)
        assert len(indices) == 2
        assert 0 in indices
        assert 1 in indices

    def test_intersection_box_and_sphere(self, grid_mesh):
        """Intersection of box and sphere."""
        box = Nodes.in_box((0, 0, 0), (2, 2, 2))  # positive octant
        sphere = Nodes.in_sphere((0, 0, 0), radius=1.1)  # near origin
        intersection = box & sphere
        indices = intersection.evaluate(grid_mesh)
        # Should be nodes in positive octant AND within radius 1.1 of origin
        for i in indices:
            x, y, z = grid_mesh.coords[i]
            assert x >= 0 and y >= 0 and z >= 0
            assert np.sqrt(x**2 + y**2 + z**2) <= 1.1

    def test_subtract_sphere_from_box(self, grid_mesh):
        """Subtract sphere from box."""
        box = Nodes.in_box((-2, -2, -2), (2, 2, 2))  # all nodes
        sphere = Nodes.in_sphere((0, 0, 0), radius=0.5)  # center only
        difference = box - sphere
        indices = difference.evaluate(grid_mesh)
        # Should exclude center node
        for i in indices:
            x, y, z = grid_mesh.coords[i]
            dist = np.sqrt(x**2 + y**2 + z**2)
            assert dist > 0.5

    def test_invert_sphere(self, sparse_mesh):
        """Invert sphere selection."""
        sphere = Nodes.in_sphere((0, 0, 0), radius=0.1)
        inverted = ~sphere
        indices = inverted.evaluate(sparse_mesh)
        # Should be all nodes except origin
        assert 0 not in indices
        assert len(indices) == sparse_mesh.n_nodes - 1


# =============================================================================
# Edge case tests
# =============================================================================


class TestGeometricQueryEdgeCases:
    """Test edge cases for geometric queries."""

    def test_empty_mesh(self):
        """Queries on empty mesh should return empty array."""
        empty_mesh = MockMesh(np.zeros((0, 3)))

        assert len(Nodes.near_point((0, 0, 0), tol=1).evaluate(empty_mesh)) == 0
        assert len(Nodes.in_sphere((0, 0, 0), radius=1).evaluate(empty_mesh)) == 0
        assert len(Nodes.in_box((-1, -1, -1), (1, 1, 1)).evaluate(empty_mesh)) == 0
        assert len(Nodes.near_line((0, 0, 0), (1, 1, 1), tol=1).evaluate(empty_mesh)) == 0

    def test_single_node_mesh(self):
        """Queries on single-node mesh."""
        single_mesh = MockMesh(np.array([[5, 5, 5]]))

        # Should find the node
        assert len(Nodes.near_point((5, 5, 5), tol=0.1).evaluate(single_mesh)) == 1
        assert len(Nodes.in_sphere((5, 5, 5), radius=0.1).evaluate(single_mesh)) == 1
        assert len(Nodes.in_box((4, 4, 4), (6, 6, 6)).evaluate(single_mesh)) == 1
        assert len(Nodes.near_line((0, 0, 0), (10, 10, 10), tol=0.1).evaluate(single_mesh)) == 1

        # Should not find the node
        assert len(Nodes.near_point((0, 0, 0), tol=0.1).evaluate(single_mesh)) == 0
        assert len(Nodes.in_sphere((0, 0, 0), radius=0.1).evaluate(single_mesh)) == 0
        assert len(Nodes.in_box((0, 0, 0), (1, 1, 1)).evaluate(single_mesh)) == 0

    def test_zero_tolerance(self, sparse_mesh):
        """Zero tolerance should only find exact matches."""
        # This tests floating point precision
        query = Nodes.near_point((0, 0, 0), tol=0)
        indices = query.evaluate(sparse_mesh)
        # Node at exact origin should still be found (within machine precision)
        assert len(indices) == 1

    def test_very_large_tolerance(self, sparse_mesh):
        """Very large tolerance should find all nodes."""
        query = Nodes.near_point((0, 0, 0), tol=1e10)
        indices = query.evaluate(sparse_mesh)
        assert len(indices) == sparse_mesh.n_nodes
