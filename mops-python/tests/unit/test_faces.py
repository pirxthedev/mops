"""Tests for Face queries and mesh face extraction.

Tests cover:
- Face extraction from elements (get_all_faces, get_boundary_faces)
- Face centroid and normal computation
- FaceQuery evaluation with various predicates
- Faces.on_boundary(), Faces.where(), Faces.on_elements()
"""

import numpy as np
import pytest

from mops import Mesh
from mops.query import Faces, FaceQuery, Elements


class TestMeshFaceExtraction:
    """Tests for mesh face extraction methods."""

    @pytest.fixture
    def tet4_mesh(self):
        """Create a single tet4 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "tet4")

    @pytest.fixture
    def hex8_mesh(self):
        """Create a unit cube hex8 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    @pytest.fixture
    def two_tet4_mesh(self):
        """Create two tet4 elements sharing a face."""
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 1.0, 0.0],  # 2
            [0.5, 0.5, 1.0],  # 3
            [0.5, 0.5, -1.0],  # 4
        ], dtype=np.float64)
        # Element 0: nodes 0,1,2,3 with face 0,1,2 (base)
        # Element 1: nodes 0,1,2,4 with face 0,1,2 shared
        elements = np.array([
            [0, 1, 2, 3],
            [0, 2, 1, 4],  # Reversed to share face with opposite orientation
        ], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "tet4")

    def test_get_all_faces_tet4(self, tet4_mesh):
        """Test getting all faces from a tet4 mesh."""
        faces = tet4_mesh.get_all_faces()

        # A single tet4 has 4 faces
        assert faces.shape == (4, 2)
        assert faces.dtype == np.int64

        # All faces are on element 0
        assert all(faces[:, 0] == 0)

        # Local face indices are 0,1,2,3
        assert set(faces[:, 1]) == {0, 1, 2, 3}

    def test_get_all_faces_hex8(self, hex8_mesh):
        """Test getting all faces from a hex8 mesh."""
        faces = hex8_mesh.get_all_faces()

        # A single hex8 has 6 faces
        assert faces.shape == (6, 2)
        assert faces.dtype == np.int64

        # All faces are on element 0
        assert all(faces[:, 0] == 0)

        # Local face indices are 0,1,2,3,4,5
        assert set(faces[:, 1]) == {0, 1, 2, 3, 4, 5}

    def test_get_boundary_faces_single_element(self, tet4_mesh):
        """Test boundary face detection for single element (all faces are boundary)."""
        boundary = tet4_mesh.get_boundary_faces()

        # For a single element, all 4 faces are boundary faces
        assert boundary.shape == (4, 2)
        assert all(boundary[:, 0] == 0)
        assert set(boundary[:, 1]) == {0, 1, 2, 3}

    def test_get_boundary_faces_shared_face(self, two_tet4_mesh):
        """Test boundary face detection with shared face."""
        boundary = two_tet4_mesh.get_boundary_faces()

        # Two tets sharing a face: 4+4 = 8 total faces, 2 shared = 6 boundary
        assert boundary.shape == (6, 2)

    def test_get_face_nodes_tet4(self, tet4_mesh):
        """Test getting face nodes for tet4 element."""
        # Face 0 of tet4 is nodes (0, 2, 1) in definition
        face_nodes = tet4_mesh.get_face_nodes(0, 0)

        assert len(face_nodes) == 3
        assert set(face_nodes) == {0, 1, 2}

    def test_get_face_nodes_hex8(self, hex8_mesh):
        """Test getting face nodes for hex8 element."""
        # Face 0 of hex8 is bottom: nodes (0, 3, 2, 1)
        face_nodes = hex8_mesh.get_face_nodes(0, 0)

        assert len(face_nodes) == 4
        assert set(face_nodes) == {0, 1, 2, 3}

    def test_get_face_centroid_tet4(self, tet4_mesh):
        """Test face centroid computation for tet4."""
        # Face 0 has nodes 0,2,1 at (0,0,0), (0,1,0), (1,0,0)
        centroid = tet4_mesh.get_face_centroid(0, 0)

        # Centroid should be average of (0,0,0), (1,0,0), (0,1,0)
        expected = np.array([1/3, 1/3, 0.0])
        np.testing.assert_array_almost_equal(centroid, expected)

    def test_get_face_centroid_hex8(self, hex8_mesh):
        """Test face centroid computation for hex8."""
        # Face 0 (bottom) has nodes at z=0
        centroid = hex8_mesh.get_face_centroid(0, 0)

        # Bottom face centroid should be at (0.5, 0.5, 0)
        expected = np.array([0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(centroid, expected)

        # Face 1 (top) has nodes at z=1
        centroid_top = hex8_mesh.get_face_centroid(0, 1)
        expected_top = np.array([0.5, 0.5, 1.0])
        np.testing.assert_array_almost_equal(centroid_top, expected_top)

    def test_get_face_normal_hex8_bottom(self, hex8_mesh):
        """Test face normal computation for hex8 bottom face."""
        normal = hex8_mesh.get_face_normal(0, 0)

        # Bottom face should have normal pointing down (-z)
        expected = np.array([0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(np.abs(normal), np.abs(expected))

    def test_get_face_normal_hex8_top(self, hex8_mesh):
        """Test face normal computation for hex8 top face."""
        normal = hex8_mesh.get_face_normal(0, 1)

        # Top face should have normal pointing up (+z)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(np.abs(normal), np.abs(expected))

    def test_get_face_normal_is_unit_vector(self, hex8_mesh):
        """Test that face normals are unit vectors."""
        for face_idx in range(6):
            normal = hex8_mesh.get_face_normal(0, face_idx)
            norm = np.linalg.norm(normal)
            assert abs(norm - 1.0) < 1e-10


class TestFacesOnBoundary:
    """Tests for Faces.on_boundary() query."""

    @pytest.fixture
    def hex8_mesh(self):
        """Create a unit cube hex8 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_on_boundary_single_hex8(self, hex8_mesh):
        """Test Faces.on_boundary() for single hex8 element."""
        query = Faces.on_boundary()
        faces = query.evaluate(hex8_mesh)

        # Single element: all 6 faces are boundary
        assert faces.shape == (6, 2)

    def test_on_boundary_predicate(self, hex8_mesh):
        """Test Faces.where(on_boundary=True) is equivalent to Faces.on_boundary()."""
        q1 = Faces.on_boundary()
        q2 = Faces.where(on_boundary=True)

        faces1 = q1.evaluate(hex8_mesh)
        faces2 = q2.evaluate(hex8_mesh)

        np.testing.assert_array_equal(faces1, faces2)


class TestFacesWhere:
    """Tests for Faces.where() with various predicates."""

    @pytest.fixture
    def hex8_mesh(self):
        """Create a unit cube hex8 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_where_z_exact(self, hex8_mesh):
        """Test selecting faces at exact z coordinate."""
        # Select faces at z=0 (bottom face)
        query = Faces.where(z=0.0)
        faces = query.evaluate(hex8_mesh)

        assert faces.shape[0] == 1
        # Verify it's face 0 (bottom)
        assert faces[0, 1] == 0

    def test_where_z_top(self, hex8_mesh):
        """Test selecting faces at z=1 (top)."""
        query = Faces.where(z=1.0)
        faces = query.evaluate(hex8_mesh)

        assert faces.shape[0] == 1
        # Verify it's face 1 (top)
        assert faces[0, 1] == 1

    def test_where_normal_up(self, hex8_mesh):
        """Test selecting faces with upward normal."""
        query = Faces.where(normal=(0, 0, 1), normal_tol=0.1)
        faces = query.evaluate(hex8_mesh)

        assert faces.shape[0] == 1
        # Verify it's the top face
        assert faces[0, 1] == 1

    def test_where_normal_down(self, hex8_mesh):
        """Test selecting faces with downward normal."""
        query = Faces.where(normal=(0, 0, -1), normal_tol=0.1)
        faces = query.evaluate(hex8_mesh)

        assert faces.shape[0] == 1
        # Verify it's the bottom face
        assert faces[0, 1] == 0

    def test_where_x_coordinate(self, hex8_mesh):
        """Test selecting faces by x centroid coordinate."""
        # Faces at x=0 or x=1 have centroid x at those values
        # Face 4 (left, -x) has centroid at x=0
        # Face 5 (right, +x) has centroid at x=1
        query_left = Faces.where(x=0.0)
        query_right = Faces.where(x=1.0)

        faces_left = query_left.evaluate(hex8_mesh)
        faces_right = query_right.evaluate(hex8_mesh)

        assert faces_left.shape[0] == 1
        assert faces_right.shape[0] == 1
        assert faces_left[0, 1] == 4  # left face
        assert faces_right[0, 1] == 5  # right face

    def test_where_z_gt(self, hex8_mesh):
        """Test z__gt predicate on face centroids."""
        # Only top face (centroid at z=1) has z > 0.9
        query = Faces.where(z__gt=0.9)
        faces = query.evaluate(hex8_mesh)

        assert faces.shape[0] == 1
        assert faces[0, 1] == 1  # top face

    def test_where_z_lt(self, hex8_mesh):
        """Test z__lt predicate on face centroids."""
        # Only bottom face (centroid at z=0) has z < 0.1
        query = Faces.where(z__lt=0.1)
        faces = query.evaluate(hex8_mesh)

        assert faces.shape[0] == 1
        assert faces[0, 1] == 0  # bottom face


class TestFacesAndWhere:
    """Tests for chaining predicates with and_where()."""

    @pytest.fixture
    def hex8_mesh(self):
        """Create a unit cube hex8 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_boundary_and_normal(self, hex8_mesh):
        """Test combining on_boundary with normal predicate."""
        query = Faces.on_boundary().and_where(normal=(0, 0, 1), normal_tol=0.1)
        faces = query.evaluate(hex8_mesh)

        assert faces.shape[0] == 1
        assert faces[0, 1] == 1  # top face

    def test_boundary_and_coordinate(self, hex8_mesh):
        """Test combining on_boundary with coordinate predicate."""
        query = Faces.on_boundary().and_where(z=0.0)
        faces = query.evaluate(hex8_mesh)

        assert faces.shape[0] == 1
        assert faces[0, 1] == 0  # bottom face


class TestFacesOnElements:
    """Tests for Faces.on_elements() query."""

    @pytest.fixture
    def multi_hex_mesh(self):
        """Create two hex8 elements stacked in z direction."""
        # Bottom cube: z from 0 to 1
        # Top cube: z from 1 to 2
        nodes = np.array([
            # Bottom cube nodes
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
            # Top cube nodes (4-7 shared, plus new top)
            [0.0, 0.0, 2.0],  # 8
            [1.0, 0.0, 2.0],  # 9
            [1.0, 1.0, 2.0],  # 10
            [0.0, 1.0, 2.0],  # 11
        ], dtype=np.float64)
        elements = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],    # Element 0 (bottom cube)
            [4, 5, 6, 7, 8, 9, 10, 11],  # Element 1 (top cube)
        ], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_on_elements_single(self, multi_hex_mesh):
        """Test selecting faces on a single element."""
        query = Faces.on_elements(Elements.by_indices([0]))
        faces = query.evaluate(multi_hex_mesh)

        # Element 0 has 6 faces
        assert faces.shape == (6, 2)
        # All should be on element 0
        assert all(faces[:, 0] == 0)

    def test_on_elements_multiple(self, multi_hex_mesh):
        """Test selecting faces on multiple elements."""
        query = Faces.on_elements(Elements.all())
        faces = query.evaluate(multi_hex_mesh)

        # 2 elements x 6 faces = 12 faces
        assert faces.shape == (12, 2)

    def test_on_elements_with_predicate(self, multi_hex_mesh):
        """Test combining on_elements with coordinate predicate."""
        # Select faces on top element with z > 1.5
        query = Faces.on_elements(Elements.by_indices([1])).and_where(z__gt=1.5)
        faces = query.evaluate(multi_hex_mesh)

        # Only the top face of element 1 should match (centroid at z=2)
        assert faces.shape[0] == 1
        assert faces[0, 0] == 1  # element 1
        assert faces[0, 1] == 1  # top face (local index)


class TestFaceQueryRepr:
    """Tests for FaceQuery string representation."""

    def test_where_repr(self):
        """Test repr for Faces.where() query."""
        query = Faces.where(z=0, on_boundary=True)
        repr_str = repr(query)

        assert "Faces.where" in repr_str
        assert "on_boundary" in repr_str or "z" in repr_str

    def test_on_boundary_repr(self):
        """Test repr for Faces.on_boundary() query."""
        query = Faces.on_boundary()
        repr_str = repr(query)

        assert "Faces.where" in repr_str
        assert "on_boundary" in repr_str

    def test_on_elements_repr(self):
        """Test repr for Faces.on_elements() query."""
        query = Faces.on_elements(Elements.all())
        repr_str = repr(query)

        assert "Faces.on_elements" in repr_str


class TestFaceQuery2D:
    """Tests for face queries on 2D elements (edges as faces)."""

    @pytest.fixture
    def tri3_mesh(self):
        """Create a single triangle mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "tri3")

    @pytest.fixture
    def quad4_mesh(self):
        """Create a single quadrilateral mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "quad4")

    def test_tri3_faces(self, tri3_mesh):
        """Test face extraction for tri3 (3 edge 'faces')."""
        faces = tri3_mesh.get_all_faces()

        assert faces.shape == (3, 2)
        assert all(faces[:, 0] == 0)
        assert set(faces[:, 1]) == {0, 1, 2}

    def test_tri3_boundary_faces(self, tri3_mesh):
        """Test boundary face extraction for tri3."""
        boundary = tri3_mesh.get_boundary_faces()

        # All edges are boundary for single triangle
        assert boundary.shape == (3, 2)

    def test_quad4_faces(self, quad4_mesh):
        """Test face extraction for quad4 (4 edge 'faces')."""
        faces = quad4_mesh.get_all_faces()

        assert faces.shape == (4, 2)
        assert all(faces[:, 0] == 0)
        assert set(faces[:, 1]) == {0, 1, 2, 3}

    def test_quad4_face_normal(self, quad4_mesh):
        """Test face normal computation for 2D element edges."""
        # Edge 0 is from (0,0,0) to (1,0,0)
        # Normal should be perpendicular in xy-plane
        normal = quad4_mesh.get_face_normal(0, 0)

        assert len(normal) == 3
        # Should be (0, -1, 0) or (0, 1, 0) depending on orientation
        assert abs(normal[0]) < 1e-10  # No x component
        assert abs(normal[2]) < 1e-10  # No z component
        assert abs(abs(normal[1]) - 1.0) < 1e-10  # Unit y component


class TestFaceQueryEdgeCases:
    """Tests for edge cases in face queries."""

    @pytest.fixture
    def hex8_mesh(self):
        """Create a unit cube hex8 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_no_matching_faces(self, hex8_mesh):
        """Test query that matches no faces."""
        # No faces at z=100
        query = Faces.where(z=100.0)
        faces = query.evaluate(hex8_mesh)

        assert faces.shape == (0, 2)

    def test_all_faces_match(self, hex8_mesh):
        """Test query that matches all faces."""
        # z__between includes all face centroids
        query = Faces.where(z__between=(-1, 2))
        faces = query.evaluate(hex8_mesh)

        # All 6 faces should match
        assert faces.shape[0] == 6

    def test_normal_tolerance(self, hex8_mesh):
        """Test that normal tolerance works correctly."""
        # With very tight tolerance, slightly off normal shouldn't match
        query_tight = Faces.where(normal=(0.01, 0, 1), normal_tol=0.001)
        faces_tight = query_tight.evaluate(hex8_mesh)

        # With loose tolerance, slightly off normal should match
        query_loose = Faces.where(normal=(0.01, 0, 1), normal_tol=0.5)
        faces_loose = query_loose.evaluate(hex8_mesh)

        # Loose tolerance should match more faces (at least top face)
        assert faces_loose.shape[0] >= faces_tight.shape[0]


class TestFaceCaching:
    """Tests for face data caching in Mesh."""

    @pytest.fixture
    def hex8_mesh(self):
        """Create a unit cube hex8 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_all_faces_cached(self, hex8_mesh):
        """Test that get_all_faces() result is cached."""
        faces1 = hex8_mesh.get_all_faces()
        faces2 = hex8_mesh.get_all_faces()

        # Should be the same object (cached)
        assert faces1 is faces2

    def test_boundary_faces_cached(self, hex8_mesh):
        """Test that get_boundary_faces() result is cached."""
        boundary1 = hex8_mesh.get_boundary_faces()
        boundary2 = hex8_mesh.get_boundary_faces()

        # Should be the same object (cached)
        assert boundary1 is boundary2

    def test_all_face_centroids_cached(self, hex8_mesh):
        """Test that get_all_face_centroids() result is cached."""
        centroids1 = hex8_mesh.get_all_face_centroids()
        centroids2 = hex8_mesh.get_all_face_centroids()

        # Should be the same object (cached)
        assert centroids1 is centroids2

    def test_all_face_normals_cached(self, hex8_mesh):
        """Test that get_all_face_normals() result is cached."""
        normals1 = hex8_mesh.get_all_face_normals()
        normals2 = hex8_mesh.get_all_face_normals()

        # Should be the same object (cached)
        assert normals1 is normals2
