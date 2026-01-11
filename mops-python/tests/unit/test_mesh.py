"""Tests for Mesh class and Gmsh integration.

Tests cover:
- Mesh.from_arrays() factory method
- Mesh.from_gmsh() Gmsh model integration
- Mesh.from_file() file loading
- Mesh properties and validation
"""

import numpy as np
import pytest

from mops import Mesh, MeshError


class TestMeshFromArrays:
    """Tests for Mesh.from_arrays() factory method."""

    def test_create_tet4_mesh(self):
        """Create a simple tet4 mesh from arrays."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)

        mesh = Mesh.from_arrays(nodes, elements, "tet4")

        assert mesh.n_nodes == 4
        assert mesh.n_elements == 1
        assert mesh.element_type == "tet4"

    def test_create_tet10_mesh(self):
        """Create a single tet10 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)

        mesh = Mesh.from_arrays(nodes, elements, "tet10")

        assert mesh.n_nodes == 10
        assert mesh.n_elements == 1
        assert mesh.element_type == "tet10"

    def test_create_hex8_mesh(self):
        """Create a unit cube hex8 mesh."""
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

        mesh = Mesh.from_arrays(nodes, elements, "hex8")

        assert mesh.n_nodes == 8
        assert mesh.n_elements == 1
        assert mesh.element_type == "hex8"

    def test_coords_property(self):
        """Test that coords returns node coordinates."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)

        mesh = Mesh.from_arrays(nodes, elements, "tet4")
        coords = mesh.coords

        assert coords.shape == (4, 3)
        np.testing.assert_array_almost_equal(coords, nodes)

    def test_unsupported_element_type(self):
        """Error for unsupported element type."""
        nodes = np.array([[0, 0, 0]], dtype=np.float64)
        elements = np.array([[0]], dtype=np.int64)

        with pytest.raises(ValueError, match="Unsupported element type"):
            Mesh.from_arrays(nodes, elements, "unsupported")

    def test_wrong_node_shape(self):
        """Error for nodes array with wrong shape."""
        nodes = np.array([[0, 0], [1, 0]], dtype=np.float64)  # Nx2 instead of Nx3
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)

        with pytest.raises(ValueError, match="nodes must be Nx3"):
            Mesh.from_arrays(nodes, elements, "tet4")

    def test_repr(self):
        """Test string representation."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)

        mesh = Mesh.from_arrays(nodes, elements, "tet4")
        repr_str = repr(mesh)

        assert "nodes=4" in repr_str
        assert "elements=1" in repr_str
        assert "tet4" in repr_str


class TestMeshFromGmsh:
    """Tests for Mesh.from_gmsh() Gmsh model integration."""

    @pytest.fixture
    def gmsh_initialized(self):
        """Initialize and finalize gmsh for tests."""
        gmsh = pytest.importorskip("gmsh")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        yield gmsh
        gmsh.finalize()

    def test_from_gmsh_box_tet4(self, gmsh_initialized):
        """Create mesh from Gmsh box with tet4 elements."""
        gmsh = gmsh_initialized
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()

        # Use first order elements (tet4)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)

        mesh = Mesh.from_gmsh(gmsh.model)

        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0
        assert mesh.element_type == "tet4"

    def test_from_gmsh_box_tet10(self, gmsh_initialized):
        """Create mesh from Gmsh box with tet10 elements."""
        gmsh = gmsh_initialized
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()

        # Use second order elements (tet10)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate(3)

        mesh = Mesh.from_gmsh(gmsh.model)

        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0
        assert mesh.element_type == "tet10"

    def test_from_gmsh_explicit_element_type(self, gmsh_initialized):
        """Specify element type explicitly when extracting from Gmsh."""
        gmsh = gmsh_initialized
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()

        # Generate both linear and quadratic elements
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)

        mesh = Mesh.from_gmsh(gmsh.model, element_type="tet4")

        assert mesh.element_type == "tet4"

    def test_from_gmsh_no_mesh_error(self, gmsh_initialized):
        """Error when no mesh has been generated."""
        gmsh = gmsh_initialized
        gmsh.model.add("empty")

        with pytest.raises(MeshError, match="No nodes found"):
            Mesh.from_gmsh(gmsh.model)

    def test_from_gmsh_invalid_element_type(self, gmsh_initialized):
        """Error for invalid element type request."""
        gmsh = gmsh_initialized
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)

        with pytest.raises(ValueError, match="Unsupported element type"):
            Mesh.from_gmsh(gmsh.model, element_type="invalid")

    def test_from_gmsh_hex8(self, gmsh_initialized):
        """Create hex8 mesh from Gmsh using transfinite meshing."""
        gmsh = gmsh_initialized
        gmsh.model.add("box")
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()

        # Get all entities
        volumes = gmsh.model.getEntities(dim=3)
        surfaces = gmsh.model.getEntities(dim=2)
        curves = gmsh.model.getEntities(dim=1)

        # Set transfinite meshing for all curves
        for dim, tag in curves:
            gmsh.model.mesh.setTransfiniteCurve(tag, 3)

        # Set transfinite meshing for all surfaces
        for dim, tag in surfaces:
            gmsh.model.mesh.setTransfiniteSurface(tag)
            gmsh.model.mesh.setRecombine(2, tag)

        # Set transfinite meshing for volume
        for dim, tag in volumes:
            gmsh.model.mesh.setTransfiniteVolume(tag)

        # First order hex elements
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)

        mesh = Mesh.from_gmsh(gmsh.model, element_type="hex8")

        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0
        assert mesh.element_type == "hex8"


class TestMeshFromFile:
    """Tests for Mesh.from_file() file loading."""

    def test_file_not_found(self):
        """Error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Mesh.from_file("/nonexistent/path.msh")

    def test_unsupported_format(self, tmp_path):
        """Error for unsupported file format."""
        invalid_file = tmp_path / "mesh.xyz"
        invalid_file.write_text("dummy content")

        with pytest.raises(MeshError, match="Unsupported file format"):
            Mesh.from_file(invalid_file)

    def test_from_msh_file(self, tmp_path):
        """Load mesh from .msh file."""
        gmsh = pytest.importorskip("gmsh")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        try:
            # Create and mesh a simple box
            gmsh.model.add("box")
            gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.model.mesh.generate(3)

            # Save to file
            msh_file = tmp_path / "box.msh"
            gmsh.write(str(msh_file))
            gmsh.clear()
        finally:
            gmsh.finalize()

        # Load from file
        mesh = Mesh.from_file(msh_file)

        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0
        assert mesh.element_type == "tet4"


class TestMeshSolveIntegration:
    """Integration tests verifying Gmsh meshes work with solver."""

    @pytest.fixture
    def gmsh_box_mesh(self):
        """Create a meshed box using Gmsh."""
        gmsh = pytest.importorskip("gmsh")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
        gmsh.model.mesh.generate(3)

        mesh = Mesh.from_gmsh(gmsh.model)

        gmsh.finalize()
        return mesh

    def test_solve_gmsh_mesh(self, gmsh_box_mesh, steel):
        """Solve a problem with mesh from Gmsh."""
        from mops import solve_simple

        mesh = gmsh_box_mesh

        # Find nodes at x=0 and x=1
        coords = mesh.coords
        x_min = coords[:, 0].min()
        x_max = coords[:, 0].max()

        fixed_nodes = np.where(np.abs(coords[:, 0] - x_min) < 1e-6)[0].astype(np.int64)
        loaded_nodes = np.where(np.abs(coords[:, 0] - x_max) < 1e-6)[0].astype(np.int64)

        assert len(fixed_nodes) > 0, "Should have fixed nodes"
        assert len(loaded_nodes) > 0, "Should have loaded nodes"

        # Apply load and solve
        load = np.array([1000.0, 0.0, 0.0], dtype=np.float64)
        results = solve_simple(
            mesh._core,  # Use the underlying Rust mesh
            steel,
            fixed_nodes,
            loaded_nodes,
            load,
        )

        # Check solution is reasonable
        displacement = results.displacement()
        assert displacement.shape == (mesh.n_nodes, 3)

        # Loaded nodes should have positive x displacement
        max_disp = results.max_displacement()
        assert max_disp > 0, "Should have positive displacement"
