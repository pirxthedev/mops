"""Tests for visualization module (mops.viz).

These tests verify the PyVista visualization integration works correctly.
Many tests require PyVista to be installed and will be skipped if not available.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import mops
from mops import Model, Mesh, Material, Nodes, Elements, Force, solve


# Skip all tests if PyVista is not available
pyvista = pytest.importorskip("pyvista")

# Check for display - screenshot tests require a display or virtual framebuffer
HAS_DISPLAY = os.environ.get("DISPLAY") is not None or os.environ.get("PYVISTA_OFF_SCREEN") == "true"


@pytest.fixture
def simple_tet4_mesh():
    """Create a simple single-element tet4 mesh for testing."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return Mesh.from_arrays(nodes, elements, "tet4")


@pytest.fixture
def simple_hex8_mesh():
    """Create a simple single-element hex8 mesh for testing."""
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
def solved_model(simple_tet4_mesh):
    """Create a solved model for testing visualization."""
    steel = Material.steel()
    model = (
        Model(simple_tet4_mesh, materials={"steel": steel})
        .assign(Elements.all(), material="steel")
        .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
        .load(Nodes.where(z__gt=0.5), Force(fz=-1000))
    )
    return solve(model), model


class TestCreatePyvistaMesh:
    """Tests for create_pyvista_mesh function."""

    def test_tet4_mesh(self, simple_tet4_mesh):
        """Test creating PyVista mesh from tet4 elements."""
        from mops.viz import create_pyvista_mesh

        pv_mesh = create_pyvista_mesh(
            simple_tet4_mesh.coords,
            simple_tet4_mesh.elements,
            simple_tet4_mesh.element_type,
        )

        assert pv_mesh.n_points == 4
        assert pv_mesh.n_cells == 1

    def test_hex8_mesh(self, simple_hex8_mesh):
        """Test creating PyVista mesh from hex8 elements."""
        from mops.viz import create_pyvista_mesh

        pv_mesh = create_pyvista_mesh(
            simple_hex8_mesh.coords,
            simple_hex8_mesh.elements,
            simple_hex8_mesh.element_type,
        )

        assert pv_mesh.n_points == 8
        assert pv_mesh.n_cells == 1

    def test_unsupported_element_type(self):
        """Test that unsupported element types raise an error."""
        from mops.viz import create_pyvista_mesh

        nodes = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        elements = np.array([[0, 1]], dtype=np.int64)

        with pytest.raises(ValueError, match="Unsupported element type"):
            create_pyvista_mesh(nodes, elements, "line2")


class TestGetFieldData:
    """Tests for _get_field_data function."""

    def test_displacement_field(self, solved_model):
        """Test extracting displacement field."""
        from mops.viz import _get_field_data

        results, _ = solved_model
        data, name, is_cell = _get_field_data(results, "displacement")

        assert data.shape == (4, 3)  # 4 nodes, 3 components
        assert name == "Displacement"
        assert is_cell is False

    def test_displacement_magnitude_field(self, solved_model):
        """Test extracting displacement magnitude field."""
        from mops.viz import _get_field_data

        results, _ = solved_model
        data, name, is_cell = _get_field_data(results, "displacement_magnitude")

        assert data.shape == (4,)  # 4 nodes, scalar
        assert name == "Displacement Magnitude"
        assert is_cell is False

    def test_von_mises_field(self, solved_model):
        """Test extracting von Mises field."""
        from mops.viz import _get_field_data

        results, _ = solved_model
        data, name, is_cell = _get_field_data(results, "von_mises")

        assert data.shape == (1,)  # 1 element
        assert name == "Von Mises Stress"
        assert is_cell is True

    def test_stress_components(self, solved_model):
        """Test extracting individual stress components."""
        from mops.viz import _get_field_data

        results, _ = solved_model

        for field in ["stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz"]:
            data, name, is_cell = _get_field_data(results, field)
            assert data.shape == (1,)  # 1 element
            assert is_cell is True

    def test_principal_stresses(self, solved_model):
        """Test extracting principal stress fields."""
        from mops.viz import _get_field_data

        results, _ = solved_model

        for i, field in enumerate(["principal_1", "principal_2", "principal_3"]):
            data, name, is_cell = _get_field_data(results, field)
            assert data.shape == (1,)
            assert f"{i+1}" in name
            assert is_cell is True

    def test_tresca_field(self, solved_model):
        """Test extracting Tresca field."""
        from mops.viz import _get_field_data

        results, _ = solved_model
        data, name, is_cell = _get_field_data(results, "tresca")

        assert data.shape == (1,)
        assert name == "Tresca Stress"
        assert is_cell is True

    def test_hydrostatic_field(self, solved_model):
        """Test extracting hydrostatic field."""
        from mops.viz import _get_field_data

        results, _ = solved_model
        data, name, is_cell = _get_field_data(results, "hydrostatic")

        assert data.shape == (1,)
        assert name == "Hydrostatic Stress"
        assert is_cell is True

    def test_unknown_field_raises(self, solved_model):
        """Test that unknown field names raise ValueError."""
        from mops.viz import _get_field_data

        results, _ = solved_model

        with pytest.raises(ValueError, match="Unknown field"):
            _get_field_data(results, "unknown_field")


class TestPlotResults:
    """Tests for plot_results function."""

    def test_plot_von_mises_returns_plotter(self, solved_model):
        """Test plotting von Mises stress returns a plotter."""
        from mops.viz import plot_results

        results, model = solved_model
        plotter = plot_results(
            results,
            model=model,
            field="von_mises",
            return_plotter=True,
            off_screen=True,
        )

        assert plotter is not None
        plotter.close()

    def test_plot_displacement_magnitude(self, solved_model):
        """Test plotting displacement magnitude."""
        from mops.viz import plot_results

        results, model = solved_model
        plotter = plot_results(
            results,
            model=model,
            field="displacement_magnitude",
            return_plotter=True,
            off_screen=True,
        )

        assert plotter is not None
        plotter.close()

    def test_plot_deformed_shape(self, solved_model):
        """Test plotting with deformed shape."""
        from mops.viz import plot_results

        results, model = solved_model
        plotter = plot_results(
            results,
            model=model,
            field="von_mises",
            deformed=True,
            scale_factor=1000,
            return_plotter=True,
            off_screen=True,
        )

        assert plotter is not None
        plotter.close()

    def test_plot_with_custom_colormap(self, solved_model):
        """Test plotting with custom colormap."""
        from mops.viz import plot_results

        results, model = solved_model
        plotter = plot_results(
            results,
            model=model,
            field="von_mises",
            cmap="plasma",
            return_plotter=True,
            off_screen=True,
        )

        assert plotter is not None
        plotter.close()

    @pytest.mark.skipif(not HAS_DISPLAY, reason="No display available for screenshot")
    def test_plot_screenshot(self, solved_model):
        """Test saving screenshot."""
        from mops.viz import plot_results

        results, model = solved_model

        with tempfile.TemporaryDirectory() as tmpdir:
            screenshot_path = Path(tmpdir) / "test_plot.png"
            plot_results(
                results,
                model=model,
                field="von_mises",
                screenshot=str(screenshot_path),
                off_screen=True,
            )

            assert screenshot_path.exists()
            assert screenshot_path.stat().st_size > 0

    def test_plot_missing_mesh_raises(self, solved_model):
        """Test that plotting without mesh data raises."""
        from mops.viz import plot_results

        results, _ = solved_model

        with pytest.raises(ValueError, match="Mesh data not available"):
            plot_results(
                results,
                model=None,  # No model, and results don't have HDF5 mesh
                field="von_mises",
                return_plotter=True,
                off_screen=True,
            )


class TestExportVtu:
    """Tests for export_vtu function."""

    def test_export_basic(self, solved_model):
        """Test basic VTU export."""
        from mops.viz import export_vtu

        results, model = solved_model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.vtu"
            export_vtu(results, output_path, model=model)

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_export_adds_vtu_extension(self, solved_model):
        """Test that .vtu extension is added if missing."""
        from mops.viz import export_vtu

        results, model = solved_model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output"  # No extension
            export_vtu(results, output_path, model=model)

            # Should add .vtu extension
            expected_path = output_path.with_suffix(".vtu")
            assert expected_path.exists()

    def test_export_specific_fields(self, solved_model):
        """Test exporting specific fields."""
        from mops.viz import export_vtu

        results, model = solved_model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.vtu"
            export_vtu(
                results,
                output_path,
                model=model,
                fields=["displacement", "von_mises"],
            )

            # Load the exported file and check fields
            mesh = pyvista.read(str(output_path))
            assert "Displacement" in mesh.point_data
            assert "Von Mises Stress" in mesh.cell_data

    def test_export_deformed(self, solved_model):
        """Test exporting deformed shape."""
        from mops.viz import export_vtu

        results, model = solved_model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_deformed.vtu"
            export_vtu(
                results,
                output_path,
                model=model,
                deformed=True,
                scale_factor=1000,
            )

            assert output_path.exists()


class TestPlotMesh:
    """Tests for plot_mesh function."""

    def test_plot_mesh_basic(self, simple_tet4_mesh):
        """Test basic mesh plotting."""
        from mops.viz import plot_mesh

        steel = Material.steel()
        model = Model(simple_tet4_mesh, materials={"steel": steel})

        plotter = plot_mesh(
            model,
            return_plotter=True,
            off_screen=True,
        )

        assert plotter is not None
        plotter.close()

    def test_plot_mesh_with_nodes(self, simple_tet4_mesh):
        """Test mesh plotting with node markers."""
        from mops.viz import plot_mesh

        steel = Material.steel()
        model = Model(simple_tet4_mesh, materials={"steel": steel})

        plotter = plot_mesh(
            model,
            show_nodes=True,
            node_size=10.0,
            return_plotter=True,
            off_screen=True,
        )

        assert plotter is not None
        plotter.close()

    @pytest.mark.skipif(not HAS_DISPLAY, reason="No display available for screenshot")
    def test_plot_mesh_screenshot(self, simple_tet4_mesh):
        """Test saving mesh screenshot."""
        from mops.viz import plot_mesh

        steel = Material.steel()
        model = Model(simple_tet4_mesh, materials={"steel": steel})

        with tempfile.TemporaryDirectory() as tmpdir:
            screenshot_path = Path(tmpdir) / "mesh_plot.png"
            plot_mesh(
                model,
                screenshot=str(screenshot_path),
                off_screen=True,
            )

            assert screenshot_path.exists()


class TestResultsPlotMethod:
    """Tests for Results.plot() method."""

    def test_results_plot_method(self, solved_model):
        """Test Results.plot() convenience method."""
        results, model = solved_model

        plotter = results.plot(
            "von_mises",
            model=model,
            return_plotter=True,
            off_screen=True,
        )

        assert plotter is not None
        plotter.close()


class TestResultsExportMethod:
    """Tests for Results.export() method."""

    def test_results_export_method(self, solved_model):
        """Test Results.export() convenience method."""
        results, model = solved_model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.vtu"
            results.export(str(output_path), model=model)

            assert output_path.exists()


class TestVtkCellTypes:
    """Tests for VTK cell type mapping."""

    def test_all_supported_types_have_vtk_mapping(self):
        """Ensure all supported element types have VTK cell type mappings."""
        from mops.viz import VTK_CELL_TYPES
        from mops.mesh import SUPPORTED_ELEMENT_TYPES

        for elem_type in SUPPORTED_ELEMENT_TYPES:
            assert elem_type in VTK_CELL_TYPES, f"Missing VTK mapping for {elem_type}"
