"""Basic tests for MOPS Python bindings."""

import numpy as np
import pytest

from mops import Material, Mesh, SolverConfig, solve_simple, solver_info, version


class TestMaterial:
    """Test Material class."""

    def test_create_material(self):
        """Test creating a custom material."""
        mat = Material("custom", e=100e9, nu=0.25)
        assert mat.name == "custom"
        assert mat.e == 100e9
        assert mat.nu == 0.25
        assert mat.rho == 0.0

    def test_create_material_with_density(self):
        """Test creating a material with density."""
        mat = Material("custom", e=100e9, nu=0.25, rho=5000.0)
        assert mat.rho == 5000.0

    def test_steel_preset(self):
        """Test steel preset material."""
        steel = Material.steel()
        assert steel.name == "steel"
        assert steel.e == 200e9
        assert steel.nu == 0.3
        assert steel.rho == 7850.0

    def test_aluminum_preset(self):
        """Test aluminum preset material."""
        aluminum = Material.aluminum()
        assert aluminum.name == "aluminum"
        assert aluminum.e == 68.9e9
        assert aluminum.nu == 0.33
        assert aluminum.rho == 2700.0

    def test_invalid_youngs_modulus(self):
        """Test that negative Young's modulus raises error."""
        with pytest.raises(ValueError):
            Material("bad", e=-100e9, nu=0.3)

    def test_invalid_poissons_ratio(self):
        """Test that invalid Poisson's ratio raises error."""
        with pytest.raises(ValueError):
            Material("bad", e=100e9, nu=0.5)
        with pytest.raises(ValueError):
            Material("bad", e=100e9, nu=-1.0)

    def test_repr(self):
        """Test material string representation."""
        mat = Material.steel()
        assert "steel" in repr(mat)
        assert "200" in repr(mat)


class TestMesh:
    """Test Mesh class."""

    def test_create_tet4_mesh(self):
        """Test creating a tet4 mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)

        mesh = Mesh(nodes, elements, "tet4")
        assert mesh.n_nodes == 4
        assert mesh.n_elements == 1

    def test_create_hex8_mesh(self):
        """Test creating a hex8 mesh (unit cube)."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

        mesh = Mesh(nodes, elements, "hex8")
        assert mesh.n_nodes == 8
        assert mesh.n_elements == 1

    def test_invalid_element_type(self):
        """Test that invalid element type raises error."""
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        elements = np.array([[0, 1, 2]], dtype=np.int64)

        with pytest.raises(ValueError):
            Mesh(nodes, elements, "invalid")

    def test_wrong_node_count(self):
        """Test that wrong node count for element type raises error."""
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # tet4 needs 4 nodes, we only provide 3
        elements = np.array([[0, 1, 2]], dtype=np.int64)

        with pytest.raises(ValueError):
            Mesh(nodes, elements, "tet4")


class TestSolverConfig:
    """Test SolverConfig class."""

    def test_default_config(self):
        """Test creating default solver config."""
        config = SolverConfig()
        # Default should work without error

    def test_direct_solver(self):
        """Test specifying direct solver."""
        config = SolverConfig(solver_type="direct")

    def test_invalid_solver_type(self):
        """Test that invalid solver type raises error."""
        with pytest.raises(ValueError):
            SolverConfig(solver_type="invalid")


class TestSolverInfo:
    """Test solver_info function."""

    def test_solver_info(self):
        """Test that solver_info returns expected keys."""
        info = solver_info()
        assert "faer_cholesky" in info
        assert info["faer_cholesky"] is True
        assert "dense_lu" in info
        assert info["dense_lu"] is True


class TestVersion:
    """Test version function."""

    def test_version(self):
        """Test that version returns a string."""
        v = version()
        assert isinstance(v, str)
        assert "." in v  # Should be in x.y.z format


class TestSolveSimple:
    """Test solve_simple function."""

    def test_solve_cantilever_tet4(self):
        """Test solving a simple cantilever beam with tet4 elements."""
        # Create a simple tet4 mesh (two tetrahedra)
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0 - fixed
            [1.0, 0.0, 0.0],  # 1 - fixed
            [0.0, 1.0, 0.0],  # 2 - fixed
            [0.0, 0.0, 1.0],  # 3 - fixed
            [1.0, 1.0, 1.0],  # 4 - loaded
        ])
        elements = np.array([
            [0, 1, 2, 4],
            [0, 1, 3, 4],
        ], dtype=np.int64)

        mesh = Mesh(nodes, elements, "tet4")
        material = Material.steel()

        # Fix nodes 0-3 (all at the base)
        constrained_nodes = np.array([0, 1, 2, 3], dtype=np.int64)

        # Apply force at node 4
        loaded_nodes = np.array([4], dtype=np.int64)
        load_vector = np.array([0.0, 0.0, -1000.0])  # 1000N downward

        # Solve
        results = solve_simple(mesh, material, constrained_nodes, loaded_nodes, load_vector)

        # Check results
        assert results.max_displacement() > 0
        disp = results.displacement()
        assert disp.shape == (5, 3)

        # Fixed nodes should have zero displacement
        for i in range(4):
            assert np.allclose(disp[i], 0.0)

        # Loaded node should have non-zero displacement
        assert not np.allclose(disp[4], 0.0)

    def test_solve_hex8(self):
        """Test solving with hex8 element."""
        # Unit cube hex8 mesh
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

        mesh = Mesh(nodes, elements, "hex8")
        material = Material.steel()

        # Fix bottom face (nodes 0-3)
        constrained_nodes = np.array([0, 1, 2, 3], dtype=np.int64)

        # Apply force at top corner
        loaded_nodes = np.array([6], dtype=np.int64)
        load_vector = np.array([1000.0, 0.0, 0.0])

        results = solve_simple(mesh, material, constrained_nodes, loaded_nodes, load_vector)

        assert results.max_displacement() > 0
        disp_mag = results.displacement_magnitude()
        assert len(disp_mag) == 8
