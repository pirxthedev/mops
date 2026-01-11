"""Basic tests for MOPS Python bindings."""

import numpy as np
import pytest

from mops import Material, Mesh, SolverConfig, solve_simple, solver_info, version

from .conftest import (
    assert_displacement_at_fixed_nodes_zero,
    assert_negative_displacement,
)


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

    def test_steel_preset(self, steel):
        """Test steel preset material."""
        assert steel.name == "steel"
        assert steel.e == 200e9
        assert steel.nu == 0.3
        assert steel.rho == 7850.0

    def test_aluminum_preset(self, aluminum):
        """Test aluminum preset material."""
        assert aluminum.name == "aluminum"
        assert aluminum.e == 68.9e9
        assert aluminum.nu == 0.33
        assert aluminum.rho == 2700.0

    def test_invalid_youngs_modulus(self):
        """Test that negative Young's modulus raises error."""
        with pytest.raises(ValueError):
            Material("bad", e=-100e9, nu=0.3)

    def test_invalid_poissons_ratio_high(self):
        """Test that Poisson's ratio >= 0.5 raises error."""
        with pytest.raises(ValueError):
            Material("bad", e=100e9, nu=0.5)

    def test_invalid_poissons_ratio_low(self):
        """Test that Poisson's ratio <= -1 raises error."""
        with pytest.raises(ValueError):
            Material("bad", e=100e9, nu=-1.0)

    def test_nearly_incompressible_material(self, nearly_incompressible):
        """Test nearly incompressible material (nu=0.499) is valid."""
        assert nearly_incompressible.nu == 0.499

    def test_repr(self, steel):
        """Test material string representation."""
        r = repr(steel)
        assert "steel" in r
        assert "2.00e11" in r or "200" in r  # E=200 GPa in scientific or standard notation


class TestMesh:
    """Test Mesh class."""

    def test_create_tet4_mesh(self, single_tet4_nodes, single_tet4_elements):
        """Test creating a tet4 mesh."""
        mesh = Mesh(single_tet4_nodes, single_tet4_elements, "tet4")
        assert mesh.n_nodes == 4
        assert mesh.n_elements == 1

    def test_single_tet4_mesh_fixture(self, single_tet4_mesh):
        """Test single tet4 mesh fixture."""
        assert single_tet4_mesh.n_nodes == 4
        assert single_tet4_mesh.n_elements == 1

    def test_two_tet4_mesh_fixture(self, two_tet4_mesh):
        """Test two tet4 elements mesh fixture."""
        assert two_tet4_mesh.n_nodes == 5
        assert two_tet4_mesh.n_elements == 2

    def test_single_tet10_mesh_fixture(self, single_tet10_mesh):
        """Test single tet10 mesh fixture."""
        assert single_tet10_mesh.n_nodes == 10
        assert single_tet10_mesh.n_elements == 1

    def test_create_hex8_mesh(self, unit_cube_hex8_nodes, unit_cube_hex8_elements):
        """Test creating a hex8 mesh (unit cube)."""
        mesh = Mesh(unit_cube_hex8_nodes, unit_cube_hex8_elements, "hex8")
        assert mesh.n_nodes == 8
        assert mesh.n_elements == 1

    def test_unit_cube_hex8_mesh_fixture(self, unit_cube_hex8_mesh):
        """Test unit cube hex8 mesh fixture."""
        assert unit_cube_hex8_mesh.n_nodes == 8
        assert unit_cube_hex8_mesh.n_elements == 1

    def test_two_hex8_mesh_fixture(self, two_hex8_mesh):
        """Test two stacked hex8 elements mesh fixture."""
        assert two_hex8_mesh.n_nodes == 12
        assert two_hex8_mesh.n_elements == 2

    def test_cantilever_tet4_fixture(self, cantilever_tet4_mesh):
        """Test cantilever tet4 mesh fixture."""
        assert cantilever_tet4_mesh.n_nodes == 5
        assert cantilever_tet4_mesh.n_elements == 2

    def test_cantilever_hex8_fixture(self, cantilever_hex8_mesh):
        """Test cantilever hex8 mesh fixture."""
        assert cantilever_hex8_mesh.n_nodes == 12
        assert cantilever_hex8_mesh.n_elements == 2

    def test_invalid_element_type(self):
        """Test that invalid element type raises error."""
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        elements = np.array([[0, 1, 2]], dtype=np.int64)

        with pytest.raises(ValueError):
            Mesh(nodes, elements, "invalid")

    def test_wrong_node_count_for_tet4(self):
        """Test that wrong node count for tet4 element type raises error."""
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        elements = np.array([[0, 1, 2]], dtype=np.int64)

        with pytest.raises(ValueError):
            Mesh(nodes, elements, "tet4")


class TestSolverConfig:
    """Test SolverConfig class."""

    def test_default_config(self):
        """Test creating default solver config."""
        config = SolverConfig()
        assert config is not None

    def test_default_solver_config_fixture(self, default_solver_config):
        """Test default solver config fixture."""
        assert default_solver_config is not None

    def test_direct_solver_config_fixture(self, direct_solver_config):
        """Test direct solver config fixture."""
        assert direct_solver_config is not None

    def test_direct_solver(self):
        """Test specifying direct solver."""
        config = SolverConfig(solver_type="direct")
        assert config is not None

    def test_invalid_solver_type(self):
        """Test that invalid solver type raises error."""
        with pytest.raises(ValueError):
            SolverConfig(solver_type="invalid")


class TestSolverInfo:
    """Test solver_info function."""

    def test_solver_info_keys(self):
        """Test that solver_info returns expected keys."""
        info = solver_info()
        assert "faer_cholesky" in info
        assert "dense_lu" in info

    def test_faer_cholesky_available(self):
        """Test that faer_cholesky solver is available."""
        info = solver_info()
        assert info["faer_cholesky"] is True

    def test_dense_lu_available(self):
        """Test that dense_lu solver is available."""
        info = solver_info()
        assert info["dense_lu"] is True


class TestVersion:
    """Test version function."""

    def test_version_is_string(self):
        """Test that version returns a string."""
        v = version()
        assert isinstance(v, str)

    def test_version_format(self):
        """Test that version is in x.y.z format."""
        v = version()
        assert "." in v


class TestSolveSimple:
    """Test solve_simple function."""

    def test_solve_two_tet4(
        self,
        two_tet4_mesh,
        steel,
        fix_first_four_nodes,
        load_last_node,
        downward_force,
    ):
        """Test solving with two tet4 elements using fixtures."""
        results = solve_simple(
            two_tet4_mesh,
            steel,
            fix_first_four_nodes,
            load_last_node,
            downward_force,
        )

        assert results.max_displacement() > 0
        disp = results.displacement()
        assert disp.shape == (5, 3)

        assert_displacement_at_fixed_nodes_zero(disp, fix_first_four_nodes)
        assert not np.allclose(disp[4], 0.0)

    def test_solve_cantilever_tet4(
        self,
        cantilever_tet4_mesh,
        steel,
        fix_first_four_nodes,
        load_last_node,
        downward_force,
    ):
        """Test solving a simple cantilever beam with tet4 elements."""
        results = solve_simple(
            cantilever_tet4_mesh,
            steel,
            fix_first_four_nodes,
            load_last_node,
            downward_force,
        )

        assert results.max_displacement() > 0
        disp = results.displacement()
        assert disp.shape == (5, 3)

        assert_displacement_at_fixed_nodes_zero(disp, fix_first_four_nodes)
        assert_negative_displacement(disp, 4, 2)  # z-direction is negative

    def test_solve_hex8_unit_cube(
        self,
        unit_cube_hex8_mesh,
        steel,
        x_direction_force,
    ):
        """Test solving with hex8 element (unit cube)."""
        # Fix bottom face (nodes 0-3)
        constrained_nodes = np.array([0, 1, 2, 3], dtype=np.int64)

        # Apply force at top corner (node 6)
        loaded_nodes = np.array([6], dtype=np.int64)

        results = solve_simple(
            unit_cube_hex8_mesh,
            steel,
            constrained_nodes,
            loaded_nodes,
            x_direction_force,
        )

        assert results.max_displacement() > 0
        disp_mag = results.displacement_magnitude()
        assert len(disp_mag) == 8

        disp = results.displacement()
        assert_displacement_at_fixed_nodes_zero(disp, constrained_nodes)

    def test_solve_two_hex8_stacked(
        self,
        two_hex8_mesh,
        steel,
        downward_force,
    ):
        """Test solving with two stacked hex8 elements."""
        # Fix bottom face (nodes 0-3)
        constrained_nodes = np.array([0, 1, 2, 3], dtype=np.int64)

        # Apply force at top corner (node 10)
        loaded_nodes = np.array([10], dtype=np.int64)

        results = solve_simple(
            two_hex8_mesh,
            steel,
            constrained_nodes,
            loaded_nodes,
            downward_force,
        )

        assert results.max_displacement() > 0
        disp = results.displacement()
        assert disp.shape == (12, 3)

        assert_displacement_at_fixed_nodes_zero(disp, constrained_nodes)

    def test_soft_material_large_displacement(
        self,
        single_tet4_mesh,
        soft_material,
    ):
        """Test that soft material produces larger displacements."""
        constrained_nodes = np.array([0, 1, 2], dtype=np.int64)
        loaded_nodes = np.array([3], dtype=np.int64)
        load = np.array([0.0, 0.0, -1000.0], dtype=np.float64)

        results = solve_simple(
            single_tet4_mesh,
            soft_material,
            constrained_nodes,
            loaded_nodes,
            load,
        )

        # Soft material should produce measurable displacement
        assert results.max_displacement() > 0


class TestSolverResults:
    """Test solver results and derived quantities."""

    def test_displacement_shape(
        self,
        two_tet4_mesh,
        steel,
        fix_first_four_nodes,
        load_last_node,
        downward_force,
    ):
        """Test displacement array has correct shape."""
        results = solve_simple(
            two_tet4_mesh,
            steel,
            fix_first_four_nodes,
            load_last_node,
            downward_force,
        )
        disp = results.displacement()
        assert disp.shape == (two_tet4_mesh.n_nodes, 3)

    def test_displacement_magnitude_shape(
        self,
        two_tet4_mesh,
        steel,
        fix_first_four_nodes,
        load_last_node,
        downward_force,
    ):
        """Test displacement magnitude array has correct shape."""
        results = solve_simple(
            two_tet4_mesh,
            steel,
            fix_first_four_nodes,
            load_last_node,
            downward_force,
        )
        disp_mag = results.displacement_magnitude()
        assert len(disp_mag) == two_tet4_mesh.n_nodes

    def test_max_displacement_positive(
        self,
        two_tet4_mesh,
        steel,
        fix_first_four_nodes,
        load_last_node,
        downward_force,
    ):
        """Test that max displacement is positive for loaded structure."""
        results = solve_simple(
            two_tet4_mesh,
            steel,
            fix_first_four_nodes,
            load_last_node,
            downward_force,
        )
        assert results.max_displacement() > 0

    def test_displacement_magnitude_matches_max(
        self,
        two_tet4_mesh,
        steel,
        fix_first_four_nodes,
        load_last_node,
        downward_force,
    ):
        """Test that max displacement equals max of displacement magnitudes."""
        results = solve_simple(
            two_tet4_mesh,
            steel,
            fix_first_four_nodes,
            load_last_node,
            downward_force,
        )
        disp_mag = results.displacement_magnitude()
        assert np.isclose(results.max_displacement(), np.max(disp_mag))
