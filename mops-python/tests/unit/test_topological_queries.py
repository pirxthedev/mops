"""Unit tests for topological query methods.

Tests the following query methods:
- Elements.attached_to(NodeQuery): Select elements where ALL nodes are in selection
- Elements.touching(NodeQuery): Select elements where ANY node is in selection
- Nodes.on_elements(ElementQuery): Select nodes on selected elements
"""

import numpy as np
import pytest

from mops.query import Elements, Nodes


class MockMesh:
    """Mock mesh for testing queries without full Rust bindings.

    Provides both node coordinates and element connectivity.
    """

    def __init__(self, coords: np.ndarray, elements: np.ndarray):
        """Create mock mesh with given coordinates and element connectivity.

        Args:
            coords: Nx3 array of node coordinates
            elements: MxK array of element connectivity (node indices)
        """
        self._coords = np.asarray(coords, dtype=np.float64)
        self._elements = np.asarray(elements, dtype=np.int64)

    @property
    def n_nodes(self) -> int:
        return self._coords.shape[0]

    @property
    def n_elements(self) -> int:
        return self._elements.shape[0]

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def elements(self) -> np.ndarray:
        return self._elements


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def single_tet_mesh() -> MockMesh:
    r"""Single tetrahedron (4 nodes, 1 element).

         3
        /|\
       / | \
      /  |  \
     0---+---2
      \  |  /
       \ | /
        \|/
         1

    Nodes:
        0: (0, 0, 0)
        1: (1, 0, 0)
        2: (0.5, 1, 0)
        3: (0.5, 0.5, 1)
    """
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return MockMesh(coords, elements)


@pytest.fixture
def two_tet_mesh() -> MockMesh:
    """Two tetrahedra sharing one face (5 nodes, 2 elements).

    Element 0: nodes [0, 1, 2, 3]
    Element 1: nodes [0, 1, 2, 4]

    Shared face: nodes [0, 1, 2]

    Nodes:
        0: (0, 0, 0)
        1: (1, 0, 0)
        2: (0.5, 1, 0)
        3: (0.5, 0.5, 1)    # apex of element 0
        4: (0.5, 0.5, -1)   # apex of element 1
    """
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        [0.5, 0.5, -1.0],
    ], dtype=np.float64)
    elements = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 4],
    ], dtype=np.int64)
    return MockMesh(coords, elements)


@pytest.fixture
def linear_tet_chain() -> MockMesh:
    """Three tetrahedra in a line (8 nodes, 3 elements).

    Element 0: nodes [0, 1, 2, 3]  (x from 0 to 1)
    Element 1: nodes [1, 4, 2, 5]  (x from 1 to 2)
    Element 2: nodes [4, 6, 2, 7]  (x from 2 to 3)

    Nodes are positioned along the x-axis with elements sharing edges.
    """
    coords = np.array([
        [0.0, 0.0, 0.0],    # 0
        [1.0, 0.0, 0.0],    # 1
        [0.5, 1.0, 0.0],    # 2
        [0.5, 0.5, 1.0],    # 3
        [2.0, 0.0, 0.0],    # 4
        [1.5, 0.5, 1.0],    # 5
        [3.0, 0.0, 0.0],    # 6
        [2.5, 0.5, 1.0],    # 7
    ], dtype=np.float64)
    elements = np.array([
        [0, 1, 2, 3],
        [1, 4, 2, 5],
        [4, 6, 2, 7],
    ], dtype=np.int64)
    return MockMesh(coords, elements)


@pytest.fixture
def isolated_elements_mesh() -> MockMesh:
    """Two completely isolated tetrahedra (8 nodes, 2 elements).

    Elements share no nodes.
    """
    coords = np.array([
        # Element 0 nodes (0-3)
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        # Element 1 nodes (4-7)
        [10.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
        [10.5, 1.0, 0.0],
        [10.5, 0.5, 1.0],
    ], dtype=np.float64)
    elements = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ], dtype=np.int64)
    return MockMesh(coords, elements)


# =============================================================================
# Elements.attached_to tests
# =============================================================================


class TestElementsAttachedTo:
    """Test Elements.attached_to(NodeQuery) - ALL nodes must match."""

    def test_attached_to_all_nodes(self, single_tet_mesh):
        """Element with all nodes selected should be included."""
        # Select all nodes (0, 1, 2, 3)
        node_query = Nodes.all()
        query = Elements.attached_to(node_query)
        indices = query.evaluate(single_tet_mesh)
        assert len(indices) == 1
        assert 0 in indices

    def test_attached_to_subset_fails(self, single_tet_mesh):
        """Element with only some nodes selected should NOT be included."""
        # Select only nodes at x=0 (just node 0)
        node_query = Nodes.where(x=0)
        query = Elements.attached_to(node_query)
        indices = query.evaluate(single_tet_mesh)
        assert len(indices) == 0

    def test_attached_to_shared_face(self, two_tet_mesh):
        """Select elements fully contained in shared face nodes."""
        # Select nodes 0, 1, 2 (shared face)
        node_query = Nodes.by_indices([0, 1, 2])
        query = Elements.attached_to(node_query)
        indices = query.evaluate(two_tet_mesh)
        # No elements are fully attached because each needs apex node too
        assert len(indices) == 0

    def test_attached_to_one_element(self, two_tet_mesh):
        """Select nodes for exactly one element."""
        # Select nodes 0, 1, 2, 3 (element 0 only)
        node_query = Nodes.by_indices([0, 1, 2, 3])
        query = Elements.attached_to(node_query)
        indices = query.evaluate(two_tet_mesh)
        assert len(indices) == 1
        assert 0 in indices

    def test_attached_to_both_elements(self, two_tet_mesh):
        """Select nodes for both elements."""
        # Select all nodes
        node_query = Nodes.all()
        query = Elements.attached_to(node_query)
        indices = query.evaluate(two_tet_mesh)
        assert len(indices) == 2
        assert 0 in indices and 1 in indices

    def test_attached_to_spatial_query(self, linear_tet_chain):
        """Attached to nodes from a spatial query."""
        # Select nodes at x <= 1 (nodes 0, 1, 2, 3 for element 0)
        node_query = Nodes.where(x__lte=1)
        query = Elements.attached_to(node_query)
        indices = query.evaluate(linear_tet_chain)
        # Only element 0 has all nodes with x <= 1
        assert len(indices) == 1
        assert 0 in indices

    def test_attached_to_empty_selection(self, single_tet_mesh):
        """Empty node selection should return no elements."""
        # Select nodes far away
        node_query = Nodes.where(x=100)
        query = Elements.attached_to(node_query)
        indices = query.evaluate(single_tet_mesh)
        assert len(indices) == 0

    def test_attached_to_repr(self):
        """Test string representation."""
        node_query = Nodes.where(x=0)
        query = Elements.attached_to(node_query)
        assert "attached_to" in repr(query)


# =============================================================================
# Elements.touching tests
# =============================================================================


class TestElementsTouching:
    """Test Elements.touching(NodeQuery) - ANY node may match."""

    def test_touching_one_node(self, single_tet_mesh):
        """Element with one node selected should be included."""
        # Select only node 0
        node_query = Nodes.by_indices([0])
        query = Elements.touching(node_query)
        indices = query.evaluate(single_tet_mesh)
        assert len(indices) == 1
        assert 0 in indices

    def test_touching_all_nodes(self, single_tet_mesh):
        """Element with all nodes selected should be included."""
        node_query = Nodes.all()
        query = Elements.touching(node_query)
        indices = query.evaluate(single_tet_mesh)
        assert len(indices) == 1
        assert 0 in indices

    def test_touching_shared_node(self, two_tet_mesh):
        """Both elements touching a shared node."""
        # Select node 0 (shared between both elements)
        node_query = Nodes.by_indices([0])
        query = Elements.touching(node_query)
        indices = query.evaluate(two_tet_mesh)
        # Both elements touch node 0
        assert len(indices) == 2
        assert 0 in indices and 1 in indices

    def test_touching_unique_apex(self, two_tet_mesh):
        """Only one element touches unique apex node."""
        # Select node 3 (apex of element 0 only)
        node_query = Nodes.by_indices([3])
        query = Elements.touching(node_query)
        indices = query.evaluate(two_tet_mesh)
        assert len(indices) == 1
        assert 0 in indices

        # Select node 4 (apex of element 1 only)
        node_query = Nodes.by_indices([4])
        query = Elements.touching(node_query)
        indices = query.evaluate(two_tet_mesh)
        assert len(indices) == 1
        assert 1 in indices

    def test_touching_spatial_query(self, linear_tet_chain):
        """Touching nodes from a spatial query."""
        # Select nodes at x=1 (node 1, shared by elements 0 and 1)
        node_query = Nodes.where(x=1)
        query = Elements.touching(node_query)
        indices = query.evaluate(linear_tet_chain)
        # Elements 0 and 1 both touch node 1
        assert len(indices) == 2
        assert 0 in indices and 1 in indices

    def test_touching_boundary(self, linear_tet_chain):
        """Touching nodes at mesh boundary."""
        # Select nodes at x=0 (only node 0, only element 0)
        node_query = Nodes.where(x=0)
        query = Elements.touching(node_query)
        indices = query.evaluate(linear_tet_chain)
        assert len(indices) == 1
        assert 0 in indices

    def test_touching_empty_selection(self, single_tet_mesh):
        """Empty node selection should return no elements."""
        node_query = Nodes.where(x=100)
        query = Elements.touching(node_query)
        indices = query.evaluate(single_tet_mesh)
        assert len(indices) == 0

    def test_touching_isolated_elements(self, isolated_elements_mesh):
        """Touching works correctly with isolated elements."""
        # Select node 0 (only element 0)
        node_query = Nodes.by_indices([0])
        query = Elements.touching(node_query)
        indices = query.evaluate(isolated_elements_mesh)
        assert len(indices) == 1
        assert 0 in indices

        # Select node 4 (only element 1)
        node_query = Nodes.by_indices([4])
        query = Elements.touching(node_query)
        indices = query.evaluate(isolated_elements_mesh)
        assert len(indices) == 1
        assert 1 in indices

    def test_touching_repr(self):
        """Test string representation."""
        node_query = Nodes.where(x=0)
        query = Elements.touching(node_query)
        assert "touching" in repr(query)


# =============================================================================
# Nodes.on_elements tests
# =============================================================================


class TestNodesOnElements:
    """Test Nodes.on_elements(ElementQuery) - nodes on selected elements."""

    def test_on_all_elements(self, single_tet_mesh):
        """Nodes on all elements should return all nodes."""
        elem_query = Elements.all()
        query = Nodes.on_elements(elem_query)
        indices = query.evaluate(single_tet_mesh)
        assert len(indices) == 4
        assert set(indices) == {0, 1, 2, 3}

    def test_on_one_element(self, two_tet_mesh):
        """Nodes on one element only."""
        # Select element 0
        elem_query = Elements.by_indices([0])
        query = Nodes.on_elements(elem_query)
        indices = query.evaluate(two_tet_mesh)
        assert set(indices) == {0, 1, 2, 3}

        # Select element 1
        elem_query = Elements.by_indices([1])
        query = Nodes.on_elements(elem_query)
        indices = query.evaluate(two_tet_mesh)
        assert set(indices) == {0, 1, 2, 4}

    def test_on_elements_shared_nodes(self, two_tet_mesh):
        """Nodes shared between elements appear once."""
        # Select both elements
        elem_query = Elements.all()
        query = Nodes.on_elements(elem_query)
        indices = query.evaluate(two_tet_mesh)
        # All 5 nodes
        assert len(indices) == 5
        assert set(indices) == {0, 1, 2, 3, 4}

    def test_on_empty_selection(self, single_tet_mesh):
        """Empty element selection returns no nodes."""
        elem_query = Elements.by_indices([])
        query = Nodes.on_elements(elem_query)
        indices = query.evaluate(single_tet_mesh)
        assert len(indices) == 0

    def test_on_elements_chain(self, linear_tet_chain):
        """Nodes on chain of elements."""
        # Select first element
        elem_query = Elements.by_indices([0])
        query = Nodes.on_elements(elem_query)
        indices = query.evaluate(linear_tet_chain)
        assert set(indices) == {0, 1, 2, 3}

        # Select last element
        elem_query = Elements.by_indices([2])
        query = Nodes.on_elements(elem_query)
        indices = query.evaluate(linear_tet_chain)
        assert set(indices) == {2, 4, 6, 7}

    def test_on_elements_isolated(self, isolated_elements_mesh):
        """Nodes on isolated elements are separate."""
        # Select element 0
        elem_query = Elements.by_indices([0])
        query = Nodes.on_elements(elem_query)
        indices = query.evaluate(isolated_elements_mesh)
        assert set(indices) == {0, 1, 2, 3}

        # Select element 1
        elem_query = Elements.by_indices([1])
        query = Nodes.on_elements(elem_query)
        indices = query.evaluate(isolated_elements_mesh)
        assert set(indices) == {4, 5, 6, 7}

    def test_on_elements_repr(self):
        """Test string representation."""
        elem_query = Elements.all()
        query = Nodes.on_elements(elem_query)
        assert "on_elements" in repr(query)


# =============================================================================
# Composition tests
# =============================================================================


class TestTopologicalQueryComposition:
    """Test composing topological queries."""

    def test_nodes_on_touching_elements(self, two_tet_mesh):
        """Get nodes on elements touching a specific node."""
        # Find elements touching node 3 (only element 0)
        touching_elems = Elements.touching(Nodes.by_indices([3]))
        # Get all nodes on those elements
        query = Nodes.on_elements(touching_elems)
        indices = query.evaluate(two_tet_mesh)
        # Should be nodes of element 0: {0, 1, 2, 3}
        assert set(indices) == {0, 1, 2, 3}

    def test_expand_and_contract(self, two_tet_mesh):
        """Expand from node to elements, then back to nodes."""
        # Start with node 0
        start_nodes = Nodes.by_indices([0])
        # Get elements touching this node
        touching = Elements.touching(start_nodes)
        # Get all nodes on those elements
        expanded = Nodes.on_elements(touching)
        indices = expanded.evaluate(two_tet_mesh)
        # Both elements touch node 0, so we get all 5 nodes
        assert set(indices) == {0, 1, 2, 3, 4}

    def test_attached_to_on_elements(self, linear_tet_chain):
        """Find elements attached to nodes of other elements."""
        # Get nodes of element 0
        elem0_nodes = Nodes.on_elements(Elements.by_indices([0]))
        # Find elements with ALL nodes in that set
        attached = Elements.attached_to(elem0_nodes)
        indices = attached.evaluate(linear_tet_chain)
        # Only element 0 itself has all nodes within its own node set
        assert len(indices) == 1
        assert 0 in indices

    def test_set_operations_on_topological(self, two_tet_mesh):
        """Set operations work with topological queries."""
        # Nodes of element 0
        elem0_nodes = Nodes.on_elements(Elements.by_indices([0]))
        # Nodes of element 1
        elem1_nodes = Nodes.on_elements(Elements.by_indices([1]))

        # Intersection should be shared nodes
        shared = elem0_nodes & elem1_nodes
        shared_indices = shared.evaluate(two_tet_mesh)
        assert set(shared_indices) == {0, 1, 2}

        # Union should be all nodes
        all_nodes = elem0_nodes | elem1_nodes
        all_indices = all_nodes.evaluate(two_tet_mesh)
        assert set(all_indices) == {0, 1, 2, 3, 4}

        # Difference should be unique nodes
        unique0 = elem0_nodes - elem1_nodes
        unique0_indices = unique0.evaluate(two_tet_mesh)
        assert set(unique0_indices) == {3}

        unique1 = elem1_nodes - elem0_nodes
        unique1_indices = unique1.evaluate(two_tet_mesh)
        assert set(unique1_indices) == {4}


# =============================================================================
# Edge case tests
# =============================================================================


class TestTopologicalQueryEdgeCases:
    """Test edge cases for topological queries."""

    def test_empty_mesh(self):
        """Queries on empty mesh should return empty array."""
        empty_mesh = MockMesh(np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64))

        # All topological queries should handle empty mesh
        result = Elements.attached_to(Nodes.all()).evaluate(empty_mesh)
        assert len(result) == 0

        result = Elements.touching(Nodes.all()).evaluate(empty_mesh)
        assert len(result) == 0

        result = Nodes.on_elements(Elements.all()).evaluate(empty_mesh)
        assert len(result) == 0

    def test_single_element_all_operations(self, single_tet_mesh):
        """All operations work on single element mesh."""
        # attached_to with all nodes
        result = Elements.attached_to(Nodes.all()).evaluate(single_tet_mesh)
        assert len(result) == 1

        # touching with any node
        result = Elements.touching(Nodes.by_indices([0])).evaluate(single_tet_mesh)
        assert len(result) == 1

        # on_elements
        result = Nodes.on_elements(Elements.all()).evaluate(single_tet_mesh)
        assert len(result) == 4
