"""Tests for Results class with HDF5 save/load functionality.

This module tests:
- Results wrapper for core solver output
- HDF5 save functionality
- HDF5 load functionality with lazy loading
- Round-trip save/load correctness
- Model metadata persistence
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mops import (
    Material,
    Mesh,
    Model,
    Nodes,
    Elements,
    Force,
    solve,
)
from mops.results import Results, FORMAT_VERSION


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    mesh = Mesh.from_arrays(nodes, elements, "tet4")

    steel = Material.steel()

    model = (
        Model(mesh, materials={"steel": steel})
        .assign(Elements.all(), material="steel")
        .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
        .load(Nodes.where(x=1), Force(fx=1000))
    )

    return model


@pytest.fixture
def solved_results(simple_model):
    """Get results from solving a simple model."""
    return solve(simple_model)


@pytest.fixture
def temp_h5_path():
    """Create a temporary file path for HDF5 testing."""
    with tempfile.NamedTemporaryFile(suffix=".mops.h5", delete=False) as f:
        path = Path(f.name)
    yield path
    # Cleanup
    if path.exists():
        path.unlink()


# =============================================================================
# Basic Results Tests
# =============================================================================


class TestResultsBasic:
    """Test basic Results functionality."""

    def test_results_from_solver(self, solved_results):
        """Results should be created from solver output."""
        assert isinstance(solved_results, Results)
        assert solved_results.n_nodes == 4
        assert solved_results.n_elements == 1

    def test_displacement_shape(self, solved_results):
        """Displacement array should have correct shape."""
        disp = solved_results.displacement()
        assert disp.shape == (4, 3)
        assert disp.dtype == np.float64

    def test_displacement_magnitude(self, solved_results):
        """Displacement magnitude should be computed correctly."""
        mag = solved_results.displacement_magnitude()
        assert mag.shape == (4,)
        assert np.all(mag >= 0)

    def test_max_displacement(self, solved_results):
        """Max displacement should be a positive scalar."""
        max_disp = solved_results.max_displacement()
        assert isinstance(max_disp, float)
        assert max_disp >= 0

    def test_stress_shape(self, solved_results):
        """Stress array should have correct shape (n_elements, 6)."""
        stress = solved_results.stress()
        assert stress.shape == (1, 6)
        assert stress.dtype == np.float64

    def test_von_mises_shape(self, solved_results):
        """Von Mises array should have correct shape (n_elements,)."""
        vm = solved_results.von_mises()
        assert vm.shape == (1,)
        assert vm.dtype == np.float64

    def test_max_von_mises(self, solved_results):
        """Max von Mises should be a positive scalar."""
        max_vm = solved_results.max_von_mises()
        assert isinstance(max_vm, float)
        assert max_vm >= 0

    def test_element_stress(self, solved_results):
        """Element stress should return 6-component vector."""
        stress = solved_results.element_stress(0)
        assert stress.shape == (6,)

    def test_element_stress_out_of_bounds(self, solved_results):
        """Element stress should raise for invalid index."""
        with pytest.raises(ValueError, match="out of bounds"):
            solved_results.element_stress(100)

    def test_element_von_mises(self, solved_results):
        """Element von Mises should return scalar."""
        vm = solved_results.element_von_mises(0)
        assert isinstance(vm, float)

    def test_element_von_mises_out_of_bounds(self, solved_results):
        """Element von Mises should raise for invalid index."""
        with pytest.raises(ValueError, match="out of bounds"):
            solved_results.element_von_mises(100)

    def test_repr(self, solved_results):
        """Results should have informative repr."""
        r = repr(solved_results)
        assert "Results" in r
        assert "n_nodes" in r
        assert "n_elements" in r


# =============================================================================
# HDF5 Save Tests
# =============================================================================


class TestResultsSave:
    """Test Results.save() functionality."""

    def test_save_creates_file(self, solved_results, temp_h5_path):
        """Save should create an HDF5 file."""
        solved_results.save(temp_h5_path)
        assert temp_h5_path.exists()

    def test_save_requires_h5py(self, solved_results, monkeypatch):
        """Save should raise ImportError if h5py not installed."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "h5py":
                raise ImportError("No module named 'h5py'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="h5py"):
            solved_results.save("/tmp/test.h5")

    def test_save_with_model(self, simple_model, solved_results, temp_h5_path):
        """Save with model should include mesh and boundary conditions."""
        import h5py

        solved_results.save(temp_h5_path, model=simple_model)

        with h5py.File(temp_h5_path, "r") as f:
            # Check mesh group exists
            assert "/mesh" in f
            assert "/mesh/nodes" in f
            assert "/mesh/elements" in f
            assert "/mesh/element_type" in f

            # Check materials group
            assert "/materials" in f
            assert "/materials/names" in f
            assert "/materials/properties" in f

            # Check model group (constraints and loads)
            assert "/model" in f
            assert "/model/constraints" in f
            assert "/model/loads" in f

    def test_save_metadata(self, solved_results, temp_h5_path):
        """Save should write correct metadata."""
        import h5py

        solved_results.save(temp_h5_path, description="Test analysis")

        with h5py.File(temp_h5_path, "r") as f:
            assert f["/metadata/format_version"][()].decode() == FORMAT_VERSION
            assert f["/metadata/analysis_type"][()].decode() == "linear_static"
            assert f["/metadata/description"][()].decode() == "Test analysis"
            assert "/metadata/mops_version" in f
            assert "/metadata/created_at" in f

    def test_save_solution_data(self, solved_results, temp_h5_path):
        """Save should write displacement and stress data."""
        import h5py

        solved_results.save(temp_h5_path)

        with h5py.File(temp_h5_path, "r") as f:
            assert "/solution/displacement" in f
            disp = f["/solution/displacement"][:]
            assert disp.shape == (4, 3)

            assert "/stress/element" in f
            stress = f["/stress/element"][:]
            assert stress.shape == (1, 6)

            assert "/stress/element_von_mises" in f
            vm = f["/stress/element_von_mises"][:]
            assert vm.shape == (1,)

    def test_save_chunked_compression(self, solved_results, temp_h5_path):
        """Save should use chunked compression."""
        import h5py

        solved_results.save(temp_h5_path)

        with h5py.File(temp_h5_path, "r") as f:
            disp = f["/solution/displacement"]
            assert disp.chunks is not None
            assert disp.compression == "gzip"


# =============================================================================
# HDF5 Load Tests
# =============================================================================


class TestResultsLoad:
    """Test Results.load() functionality."""

    def test_load_returns_results(self, solved_results, temp_h5_path):
        """Load should return a Results object."""
        solved_results.save(temp_h5_path)

        loaded = Results.load(temp_h5_path)
        try:
            assert isinstance(loaded, Results)
        finally:
            loaded.close()

    def test_load_file_not_found(self):
        """Load should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            Results.load("/nonexistent/path.mops.h5")

    def test_load_lazy_mode(self, solved_results, temp_h5_path):
        """Loaded results should be in lazy mode."""
        solved_results.save(temp_h5_path)

        loaded = Results.load(temp_h5_path)
        try:
            assert loaded._lazy is True
            assert loaded._h5file is not None
        finally:
            loaded.close()

    def test_load_metadata(self, solved_results, temp_h5_path):
        """Loaded results should have correct metadata."""
        solved_results.save(temp_h5_path, description="Test description")

        loaded = Results.load(temp_h5_path)
        try:
            assert loaded.analysis_type == "linear_static"
            assert loaded.description == "Test description"
            assert loaded.created_at is not None
        finally:
            loaded.close()

    def test_load_context_manager(self, solved_results, temp_h5_path):
        """Results.load() should work as context manager."""
        solved_results.save(temp_h5_path)

        with Results.load(temp_h5_path) as loaded:
            disp = loaded.displacement()
            assert disp.shape == (4, 3)

        # File should be closed after context
        assert loaded._h5file is None

    def test_close_idempotent(self, solved_results, temp_h5_path):
        """Calling close() multiple times should be safe."""
        solved_results.save(temp_h5_path)

        loaded = Results.load(temp_h5_path)
        loaded.close()
        loaded.close()  # Should not raise


# =============================================================================
# Round-trip Tests
# =============================================================================


class TestResultsRoundTrip:
    """Test save/load round-trip preserves data."""

    def test_displacement_roundtrip(self, solved_results, temp_h5_path):
        """Displacement data should survive round-trip."""
        original_disp = solved_results.displacement()

        solved_results.save(temp_h5_path)

        with Results.load(temp_h5_path) as loaded:
            loaded_disp = loaded.displacement()
            np.testing.assert_allclose(loaded_disp, original_disp, rtol=1e-10)

    def test_stress_roundtrip(self, solved_results, temp_h5_path):
        """Stress data should survive round-trip."""
        original_stress = solved_results.stress()

        solved_results.save(temp_h5_path)

        with Results.load(temp_h5_path) as loaded:
            loaded_stress = loaded.stress()
            np.testing.assert_allclose(loaded_stress, original_stress, rtol=1e-10)

    def test_von_mises_roundtrip(self, solved_results, temp_h5_path):
        """Von Mises data should survive round-trip."""
        original_vm = solved_results.von_mises()

        solved_results.save(temp_h5_path)

        with Results.load(temp_h5_path) as loaded:
            loaded_vm = loaded.von_mises()
            np.testing.assert_allclose(loaded_vm, original_vm, rtol=1e-10)

    def test_max_displacement_roundtrip(self, solved_results, temp_h5_path):
        """Max displacement should match after round-trip."""
        original_max = solved_results.max_displacement()

        solved_results.save(temp_h5_path)

        with Results.load(temp_h5_path) as loaded:
            loaded_max = loaded.max_displacement()
            np.testing.assert_allclose(loaded_max, original_max, rtol=1e-10)

    def test_max_von_mises_roundtrip(self, solved_results, temp_h5_path):
        """Max von Mises should match after round-trip."""
        original_max = solved_results.max_von_mises()

        solved_results.save(temp_h5_path)

        with Results.load(temp_h5_path) as loaded:
            loaded_max = loaded.max_von_mises()
            np.testing.assert_allclose(loaded_max, original_max, rtol=1e-10)

    def test_n_nodes_roundtrip(self, simple_model, solved_results, temp_h5_path):
        """Node count should match after round-trip with model."""
        solved_results.save(temp_h5_path, model=simple_model)

        with Results.load(temp_h5_path) as loaded:
            assert loaded.n_nodes == solved_results.n_nodes

    def test_n_elements_roundtrip(self, simple_model, solved_results, temp_h5_path):
        """Element count should match after round-trip with model."""
        solved_results.save(temp_h5_path, model=simple_model)

        with Results.load(temp_h5_path) as loaded:
            assert loaded.n_elements == solved_results.n_elements

    def test_element_type_roundtrip(self, simple_model, solved_results, temp_h5_path):
        """Element type should match after round-trip with model."""
        solved_results.save(temp_h5_path, model=simple_model)

        with Results.load(temp_h5_path) as loaded:
            assert loaded.element_type == "tet4"


# =============================================================================
# Mesh Metadata Tests (with model)
# =============================================================================


class TestResultsMeshMetadata:
    """Test mesh metadata persistence."""

    def test_mesh_nodes_saved(self, simple_model, solved_results, temp_h5_path):
        """Mesh nodes should be saved correctly."""
        import h5py

        solved_results.save(temp_h5_path, model=simple_model)

        with h5py.File(temp_h5_path, "r") as f:
            nodes = f["/mesh/nodes"][:]
            expected = simple_model.mesh.coords
            np.testing.assert_allclose(nodes, expected)

    def test_mesh_elements_saved(self, simple_model, solved_results, temp_h5_path):
        """Mesh elements should be saved correctly."""
        import h5py

        solved_results.save(temp_h5_path, model=simple_model)

        with h5py.File(temp_h5_path, "r") as f:
            elements = f["/mesh/elements"][:]
            expected = simple_model.mesh.elements
            np.testing.assert_array_equal(elements, expected)

    def test_mesh_bounds_saved(self, simple_model, solved_results, temp_h5_path):
        """Mesh bounds should be saved correctly."""
        import h5py

        solved_results.save(temp_h5_path, model=simple_model)

        with h5py.File(temp_h5_path, "r") as f:
            bounds = f["/mesh/bounds"][:]
            assert bounds.shape == (2, 3)
            # Min should be [0, 0, 0], max should be [1, 1, 1]
            np.testing.assert_allclose(bounds[0], [0, 0, 0])
            np.testing.assert_allclose(bounds[1], [1, 1, 1])


# =============================================================================
# Material Metadata Tests
# =============================================================================


class TestResultsMaterialMetadata:
    """Test material metadata persistence."""

    def test_material_names_saved(self, simple_model, solved_results, temp_h5_path):
        """Material names should be saved."""
        import h5py

        solved_results.save(temp_h5_path, model=simple_model)

        with h5py.File(temp_h5_path, "r") as f:
            names = [n.decode() for n in f["/materials/names"][:]]
            assert "steel" in names

    def test_material_properties_saved(self, simple_model, solved_results, temp_h5_path):
        """Material properties should be saved correctly."""
        import h5py

        steel = Material.steel()
        solved_results.save(temp_h5_path, model=simple_model)

        with h5py.File(temp_h5_path, "r") as f:
            props = f["/materials/properties"][:]
            assert props.shape[1] == 4  # [E, nu, rho, alpha]
            # First material should be steel
            np.testing.assert_allclose(props[0, 0], steel.e, rtol=1e-10)
            np.testing.assert_allclose(props[0, 1], steel.nu, rtol=1e-10)
            np.testing.assert_allclose(props[0, 2], steel.rho, rtol=1e-10)


# =============================================================================
# Constraint Metadata Tests
# =============================================================================


class TestResultsConstraintMetadata:
    """Test constraint metadata persistence."""

    def test_constraints_saved(self, simple_model, solved_results, temp_h5_path):
        """Constraints should be saved."""
        import h5py

        solved_results.save(temp_h5_path, model=simple_model)

        with h5py.File(temp_h5_path, "r") as f:
            assert "/model/constraints/node_indices" in f
            assert "/model/constraints/dof_mask" in f
            assert "/model/constraints/prescribed_values" in f

    def test_constraint_nodes(self, simple_model, solved_results, temp_h5_path):
        """Constrained node indices should be saved correctly."""
        import h5py

        solved_results.save(temp_h5_path, model=simple_model)

        with h5py.File(temp_h5_path, "r") as f:
            node_indices = f["/model/constraints/node_indices"][:]
            # Node 0 (at x=0) should be constrained
            assert 0 in node_indices


# =============================================================================
# Load Metadata Tests
# =============================================================================


class TestResultsLoadMetadata:
    """Test load metadata persistence."""

    def test_loads_saved(self, simple_model, solved_results, temp_h5_path):
        """Loads should be saved."""
        import h5py

        solved_results.save(temp_h5_path, model=simple_model)

        with h5py.File(temp_h5_path, "r") as f:
            assert "/model/loads/node_indices" in f
            assert "/model/loads/nodal_forces" in f

    def test_load_nodes(self, simple_model, solved_results, temp_h5_path):
        """Loaded node indices should be saved correctly."""
        import h5py

        solved_results.save(temp_h5_path, model=simple_model)

        with h5py.File(temp_h5_path, "r") as f:
            node_indices = f["/model/loads/node_indices"][:]
            # Node 1 (at x=1) should have load
            assert 1 in node_indices


# =============================================================================
# Edge Cases
# =============================================================================


class TestResultsEdgeCases:
    """Test edge cases and error handling."""

    def test_save_without_model(self, solved_results, temp_h5_path):
        """Save without model should still work (no mesh/BC metadata)."""
        import h5py

        solved_results.save(temp_h5_path)

        with h5py.File(temp_h5_path, "r") as f:
            # Solution should exist
            assert "/solution/displacement" in f
            # Mesh should not exist (no model provided)
            assert "/mesh" not in f

    def test_empty_description(self, solved_results, temp_h5_path):
        """Empty description should be saved correctly."""
        import h5py

        solved_results.save(temp_h5_path, description="")

        with h5py.File(temp_h5_path, "r") as f:
            desc = f["/metadata/description"][()].decode()
            assert desc == ""

    def test_caching(self, solved_results, temp_h5_path):
        """Data should be cached after first access."""
        solved_results.save(temp_h5_path)

        with Results.load(temp_h5_path) as loaded:
            # First access - loads from file
            disp1 = loaded.displacement()
            # Second access - should return cached
            disp2 = loaded.displacement()

            assert disp1 is disp2  # Same object (cached)

    def test_repr_lazy(self, solved_results, temp_h5_path):
        """Repr should indicate lazy loading mode."""
        solved_results.save(temp_h5_path)

        with Results.load(temp_h5_path) as loaded:
            r = repr(loaded)
            assert "file" in r  # source=file for lazy loading


# =============================================================================
# Derived Quantities Tests
# =============================================================================


class TestResultsDerivedQuantities:
    """Test derived quantity computation on Results."""

    def test_principal_stresses_shape(self, solved_results):
        """Principal stresses should have shape (n_elements, 3)."""
        principals = solved_results.principal_stresses()
        assert principals.shape == (1, 3)
        assert principals.dtype == np.float64

    def test_principal_stresses_ordering(self, solved_results):
        """Principal stresses should be in descending order."""
        principals = solved_results.principal_stresses()
        assert principals[0, 0] >= principals[0, 1] >= principals[0, 2]

    def test_tresca_shape(self, solved_results):
        """Tresca stress should have shape (n_elements,)."""
        tresca = solved_results.tresca()
        assert tresca.shape == (1,)
        assert tresca.dtype == np.float64

    def test_max_tresca(self, solved_results):
        """Max Tresca should be a positive scalar."""
        max_tresca = solved_results.max_tresca()
        assert isinstance(max_tresca, float)
        assert max_tresca >= 0

    def test_tresca_relation_to_principal(self, solved_results):
        """Tresca should equal σ1 - σ3."""
        principals = solved_results.principal_stresses()
        tresca = solved_results.tresca()
        expected = principals[:, 0] - principals[:, 2]
        np.testing.assert_allclose(tresca, expected, rtol=1e-10)

    def test_max_shear_stress(self, solved_results):
        """Max shear should be half of Tresca."""
        tresca = solved_results.tresca()
        max_shear = solved_results.max_shear_stress()
        np.testing.assert_allclose(max_shear, tresca / 2.0, rtol=1e-10)

    def test_hydrostatic_stress_shape(self, solved_results):
        """Hydrostatic stress should have shape (n_elements,)."""
        hydro = solved_results.hydrostatic_stress()
        assert hydro.shape == (1,)
        assert hydro.dtype == np.float64

    def test_pressure_relation(self, solved_results):
        """Pressure should be negative hydrostatic stress."""
        hydro = solved_results.hydrostatic_stress()
        p = solved_results.pressure()
        np.testing.assert_allclose(p, -hydro, rtol=1e-10)

    def test_stress_intensity_shape(self, solved_results):
        """Stress intensity should have shape (n_elements,)."""
        intensity = solved_results.stress_intensity()
        assert intensity.shape == (1,)
        assert intensity.dtype == np.float64

    def test_derived_caching(self, solved_results):
        """Derived quantities should be cached."""
        # First call computes
        principals1 = solved_results.principal_stresses()
        # Second call should return cached
        principals2 = solved_results.principal_stresses()
        assert principals1 is principals2


class TestResultsDerivedRoundTrip:
    """Test derived quantities survive HDF5 round-trip."""

    def test_principal_stresses_roundtrip(self, simple_model, solved_results, temp_h5_path):
        """Principal stresses should survive round-trip."""
        original = solved_results.principal_stresses()

        solved_results.save(temp_h5_path, model=simple_model)

        with Results.load(temp_h5_path) as loaded:
            loaded_principals = loaded.principal_stresses()
            np.testing.assert_allclose(loaded_principals, original, rtol=1e-10)

    def test_tresca_roundtrip(self, simple_model, solved_results, temp_h5_path):
        """Tresca stress should survive round-trip."""
        original = solved_results.tresca()

        solved_results.save(temp_h5_path, model=simple_model)

        with Results.load(temp_h5_path) as loaded:
            loaded_tresca = loaded.tresca()
            np.testing.assert_allclose(loaded_tresca, original, rtol=1e-10)

    def test_hydrostatic_roundtrip(self, simple_model, solved_results, temp_h5_path):
        """Hydrostatic stress should survive round-trip."""
        original = solved_results.hydrostatic_stress()

        solved_results.save(temp_h5_path, model=simple_model)

        with Results.load(temp_h5_path) as loaded:
            loaded_hydro = loaded.hydrostatic_stress()
            np.testing.assert_allclose(loaded_hydro, original, rtol=1e-10)

    def test_stress_intensity_roundtrip(self, simple_model, solved_results, temp_h5_path):
        """Stress intensity should survive round-trip."""
        original = solved_results.stress_intensity()

        solved_results.save(temp_h5_path, model=simple_model)

        with Results.load(temp_h5_path) as loaded:
            loaded_intensity = loaded.stress_intensity()
            np.testing.assert_allclose(loaded_intensity, original, rtol=1e-10)

    def test_derived_hdf5_caching(self, simple_model, solved_results, temp_h5_path):
        """Derived quantities should be read from HDF5 cache."""
        import h5py

        solved_results.save(temp_h5_path, model=simple_model)

        # Verify derived quantities are stored in HDF5
        with h5py.File(temp_h5_path, "r") as f:
            assert "/stress/element_principal" in f
            assert "/stress/element_tresca" in f
            assert "/stress/element_hydrostatic" in f
            assert "/stress/element_intensity" in f
