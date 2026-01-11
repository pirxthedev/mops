"""Results class with HDF5 save/load functionality.

This module provides the Results class for storing and persisting FEA analysis results.
Results can be saved to HDF5 format for efficient storage and lazy loading of large datasets.

Example::

    from mops import Model, solve

    # Solve model
    results = solve(model)

    # Save to HDF5
    results.save("analysis.mops.h5", model=model, description="Cantilever analysis")

    # Load results (lazy loading - file stays open)
    loaded = Results.load("analysis.mops.h5")
    print(loaded.max_displacement())

    # Close when done
    loaded.close()

    # Or use context manager
    with Results.load("analysis.mops.h5") as results:
        disp = results.displacement()
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import mops

if TYPE_CHECKING:
    from mops._core import Results as CoreResults
    from mops.model import Model

# HDF5 format version
FORMAT_VERSION = "1.0"


class Results:
    """FEA solution results with HDF5 persistence.

    This class wraps the core solver results and adds:
    - HDF5 save/load functionality
    - Lazy loading for large result files
    - Model metadata storage for reproducibility

    Results can be created from solver output or loaded from HDF5 files.
    When loaded from files, data is accessed lazily to minimize memory usage.

    Attributes:
        n_nodes: Number of nodes in the mesh.
        n_elements: Number of elements in the mesh.

    Example::

        # From solver
        results = solve(model)
        results.save("output.mops.h5", model=model)

        # From file
        results = Results.load("output.mops.h5")
        disp = results.displacement()  # Loads on demand
    """

    def __init__(
        self,
        core_results: "CoreResults | None" = None,
        *,
        _hdf5_file: object | None = None,
        _lazy: bool = False,
    ) -> None:
        """Initialize Results from core solver output or HDF5 file.

        This constructor is typically not called directly. Instead use:
        - solve(model) to get results from solver
        - Results.load(path) to load from HDF5 file

        Args:
            core_results: Results object from Rust solver.
            _hdf5_file: Internal h5py File handle for lazy loading.
            _lazy: Internal flag indicating lazy loading mode.
        """
        self._core = core_results
        self._h5file = _hdf5_file
        self._lazy = _lazy

        # Cached data for lazy loading
        self._displacement_cache: NDArray[np.float64] | None = None
        self._stress_cache: NDArray[np.float64] | None = None
        self._von_mises_cache: NDArray[np.float64] | None = None

        # Metadata (populated on save or load)
        self._n_nodes: int | None = None
        self._n_elements: int | None = None
        self._element_type: str | None = None
        self._analysis_type: str = "linear_static"
        self._description: str = ""
        self._created_at: str | None = None
        self._mops_version: str | None = None

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the mesh."""
        if self._n_nodes is not None:
            return self._n_nodes
        if self._lazy and self._h5file is not None:
            # Prefer mesh nodes if available, fall back to displacement shape
            if "/mesh/nodes" in self._h5file:
                return self._h5file["/mesh/nodes"].shape[0]
            return self._h5file["/solution/displacement"].shape[0]
        if self._core is not None:
            # Get from displacement shape
            return self._core.displacement().shape[0]
        raise RuntimeError("Results not initialized")

    @property
    def n_elements(self) -> int:
        """Number of elements in the mesh."""
        if self._n_elements is not None:
            return self._n_elements
        if self._lazy and self._h5file is not None:
            # Prefer mesh elements if available, fall back to stress shape
            if "/mesh/elements" in self._h5file:
                return self._h5file["/mesh/elements"].shape[0]
            return self._h5file["/stress/element"].shape[0]
        if self._core is not None:
            return self._core.stress().shape[0]
        raise RuntimeError("Results not initialized")

    @property
    def element_type(self) -> str | None:
        """Element type of the mesh."""
        if self._element_type is not None:
            return self._element_type
        if self._lazy and self._h5file is not None:
            return self._h5file["/mesh/element_type"][()].decode()
        return None

    @property
    def analysis_type(self) -> str:
        """Type of analysis performed."""
        return self._analysis_type

    @property
    def description(self) -> str:
        """User-provided description."""
        return self._description

    @property
    def created_at(self) -> str | None:
        """ISO 8601 timestamp when results were created."""
        return self._created_at

    def displacement(self) -> NDArray[np.float64]:
        """Get nodal displacements.

        Returns:
            (n_nodes, 3) array of displacements [ux, uy, uz] per node.
        """
        if self._displacement_cache is not None:
            return self._displacement_cache

        if self._lazy and self._h5file is not None:
            self._displacement_cache = self._h5file["/solution/displacement"][:]
            return self._displacement_cache

        if self._core is not None:
            # Get displacement from core and cache it
            self._displacement_cache = np.asarray(self._core.displacement())
            return self._displacement_cache

        raise RuntimeError("Results not initialized")

    def displacement_magnitude(self) -> NDArray[np.float64]:
        """Get displacement magnitude per node.

        Returns:
            (n_nodes,) array of displacement magnitudes.
        """
        disp = self.displacement()
        return np.linalg.norm(disp, axis=1)

    def max_displacement(self) -> float:
        """Get maximum displacement magnitude.

        Returns:
            Maximum displacement magnitude across all nodes.
        """
        return float(np.max(self.displacement_magnitude()))

    def stress(self) -> NDArray[np.float64]:
        """Get element stress tensors.

        Returns:
            (n_elements, 6) array of stress tensors in Voigt notation:
            [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]
        """
        if self._stress_cache is not None:
            return self._stress_cache

        if self._lazy and self._h5file is not None:
            self._stress_cache = self._h5file["/stress/element"][:]
            return self._stress_cache

        if self._core is not None:
            self._stress_cache = np.asarray(self._core.stress())
            return self._stress_cache

        raise RuntimeError("Results not initialized")

    def von_mises(self) -> NDArray[np.float64]:
        """Get von Mises stress per element.

        Returns:
            (n_elements,) array of von Mises stress values.
        """
        if self._von_mises_cache is not None:
            return self._von_mises_cache

        if self._lazy and self._h5file is not None:
            self._von_mises_cache = self._h5file["/stress/element_von_mises"][:]
            return self._von_mises_cache

        if self._core is not None:
            self._von_mises_cache = np.asarray(self._core.von_mises())
            return self._von_mises_cache

        raise RuntimeError("Results not initialized")

    def max_von_mises(self) -> float:
        """Get maximum von Mises stress.

        Returns:
            Maximum von Mises stress across all elements.
        """
        return float(np.max(self.von_mises()))

    def element_stress(self, element_id: int) -> NDArray[np.float64]:
        """Get stress tensor for a specific element.

        Args:
            element_id: Element index.

        Returns:
            (6,) array of stress tensor components in Voigt notation.
        """
        stress = self.stress()
        if element_id < 0 or element_id >= len(stress):
            raise ValueError(f"Element index {element_id} out of bounds (n_elements={len(stress)})")
        return stress[element_id]

    def element_von_mises(self, element_id: int) -> float:
        """Get von Mises stress for a specific element.

        Args:
            element_id: Element index.

        Returns:
            Von Mises stress value.
        """
        vm = self.von_mises()
        if element_id < 0 or element_id >= len(vm):
            raise ValueError(f"Element index {element_id} out of bounds (n_elements={len(vm)})")
        return float(vm[element_id])

    def save(
        self,
        path: str | Path,
        *,
        model: "Model | None" = None,
        description: str = "",
    ) -> None:
        """Save results to HDF5 format.

        Creates a self-contained .mops.h5 file containing:
        - Solution data (displacements, stresses)
        - Mesh geometry (if model provided)
        - Material properties (if model provided)
        - Constraints and loads (if model provided)
        - Metadata (version, timestamp, description)

        Args:
            path: Output file path (should end in .mops.h5).
            model: Optional model to include mesh and boundary conditions.
            description: Optional description for metadata.

        Example::

            results = solve(model)
            results.save("bracket_analysis.mops.h5", model=model, description="Bracket stress analysis")
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py package required for HDF5 save/load. "
                "Install with: pip install h5py"
            )

        path = Path(path)

        with h5py.File(path, "w") as f:
            # Metadata group
            meta = f.create_group("metadata")
            meta.create_dataset("format_version", data=FORMAT_VERSION)
            meta.create_dataset("mops_version", data=mops.__version__)
            meta.create_dataset("created_at", data=datetime.now().isoformat())
            meta.create_dataset("analysis_type", data=self._analysis_type)
            meta.create_dataset("description", data=description)

            # Mesh group (if model provided)
            if model is not None:
                mesh = model.mesh
                mesh_grp = f.create_group("mesh")
                _create_chunked_dataset(
                    mesh_grp, "nodes", mesh.coords,
                    chunks=(min(1000, mesh.n_nodes), 3),
                )
                _create_chunked_dataset(
                    mesh_grp, "elements", mesh.elements,
                    chunks=(min(1000, mesh.n_elements), mesh.elements.shape[1]),
                )
                mesh_grp.create_dataset("element_type", data=mesh.element_type)

                # Compute and store bounds
                coords = mesh.coords
                bounds = np.array([
                    coords.min(axis=0),
                    coords.max(axis=0),
                ])
                mesh_grp.create_dataset("bounds", data=bounds)

                # Materials group
                self._save_materials(f, model)

                # Model group (constraints and loads)
                self._save_model_definition(f, model)

                # Components group
                self._save_components(f, model)

            # Solution group
            sol = f.create_group("solution")
            disp = self.displacement()
            _create_chunked_dataset(
                sol, "displacement", disp,
                chunks=(min(1000, len(disp)), 3),
            )

            # Solver metadata (placeholder - to be enhanced with actual solver stats)
            solver_grp = sol.create_group("solver")
            solver_grp.create_dataset("type", data="cholesky")
            solver_grp.create_dataset("iterations", data=0)
            solver_grp.create_dataset("residual_norm", data=0.0)
            solver_grp.create_dataset("factorization_time", data=0.0)
            solver_grp.create_dataset("solve_time", data=0.0)
            solver_grp.create_dataset("peak_memory_mb", data=0.0)

            # Stress group
            stress_grp = f.create_group("stress")
            stress_data = self.stress()
            _create_chunked_dataset(
                stress_grp, "element", stress_data,
                chunks=(min(1000, len(stress_data)), 6),
            )
            vm_data = self.von_mises()
            _create_chunked_dataset(
                stress_grp, "element_von_mises", vm_data,
                chunks=(min(10000, len(vm_data)),),
            )

    def _save_materials(self, f: "h5py.File", model: "Model") -> None:
        """Save materials group to HDF5 file."""
        materials = model.materials
        if not materials:
            return

        mat_grp = f.create_group("materials")

        # Material names
        names = list(materials.keys())
        mat_grp.create_dataset("names", data=names)

        # Material properties: [E, nu, rho, alpha]
        props = np.zeros((len(names), 4), dtype=np.float64)
        for i, name in enumerate(names):
            mat = materials[name]
            props[i, 0] = mat.e
            props[i, 1] = mat.nu
            props[i, 2] = mat.rho
            props[i, 3] = 0.0  # alpha (thermal expansion) - not yet supported
        mat_grp.create_dataset("properties", data=props)

        # Element material IDs (all elements get first material for now)
        # TODO: Support multiple materials per element region
        n_elements = model.mesh.n_elements
        element_ids = np.zeros(n_elements, dtype=np.int32)
        mat_grp.create_dataset("element_material_ids", data=element_ids)

    def _save_model_definition(self, f: "h5py.File", model: "Model") -> None:
        """Save model constraints and loads to HDF5 file."""
        model_grp = f.create_group("model")

        # Constraints subgroup
        constraints_grp = model_grp.create_group("constraints")

        # Evaluate constraints to get actual node indices
        state = model._state
        dof_map = {"ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5}

        # Collect constraint data
        constrained_nodes = {}  # node -> (dof_mask, values)

        for query, dofs, value in state.constraints:
            try:
                indices = query.evaluate(model.mesh, state.components)
                for node_idx in indices:
                    if node_idx not in constrained_nodes:
                        constrained_nodes[node_idx] = (
                            np.zeros(6, dtype=bool),
                            np.zeros(6, dtype=np.float64),
                        )
                    mask, vals = constrained_nodes[node_idx]
                    for dof_name in dofs:
                        if dof_name in dof_map:
                            dof_idx = dof_map[dof_name]
                            mask[dof_idx] = True
                            vals[dof_idx] = value
            except Exception:
                # Skip constraints that can't be evaluated (e.g., missing mesh)
                pass

        if constrained_nodes:
            node_indices = np.array(sorted(constrained_nodes.keys()), dtype=np.int64)
            dof_mask = np.array([constrained_nodes[n][0] for n in node_indices], dtype=bool)
            prescribed_values = np.array([constrained_nodes[n][1] for n in node_indices], dtype=np.float64)

            constraints_grp.create_dataset("node_indices", data=node_indices)
            constraints_grp.create_dataset("dof_mask", data=dof_mask)
            constraints_grp.create_dataset("prescribed_values", data=prescribed_values)
        else:
            # Empty arrays with correct shape
            constraints_grp.create_dataset("node_indices", data=np.array([], dtype=np.int64))
            constraints_grp.create_dataset("dof_mask", data=np.zeros((0, 6), dtype=bool))
            constraints_grp.create_dataset("prescribed_values", data=np.zeros((0, 6), dtype=np.float64))

        # Loads subgroup
        loads_grp = model_grp.create_group("loads")

        # Collect nodal forces
        loaded_nodes = {}  # node -> force_vector

        for query, load in state.loads:
            try:
                indices = query.evaluate(model.mesh, state.components)
                if hasattr(load, "fx"):  # Force load
                    force = np.array([load.fx, load.fy, load.fz, 0.0, 0.0, 0.0])
                    for node_idx in indices:
                        if node_idx not in loaded_nodes:
                            loaded_nodes[node_idx] = np.zeros(6, dtype=np.float64)
                        loaded_nodes[node_idx] += force
            except Exception:
                pass

        if loaded_nodes:
            node_indices = np.array(sorted(loaded_nodes.keys()), dtype=np.int64)
            nodal_forces = np.array([loaded_nodes[n] for n in node_indices], dtype=np.float64)

            loads_grp.create_dataset("node_indices", data=node_indices)
            loads_grp.create_dataset("nodal_forces", data=nodal_forces)
        else:
            loads_grp.create_dataset("node_indices", data=np.array([], dtype=np.int64))
            loads_grp.create_dataset("nodal_forces", data=np.zeros((0, 6), dtype=np.float64))

        # Pressure loads (placeholder - not fully supported yet)
        loads_grp.create_dataset("face_elements", data=np.array([], dtype=np.int64))
        loads_grp.create_dataset("face_local_ids", data=np.array([], dtype=np.int32))
        loads_grp.create_dataset("face_pressures", data=np.array([], dtype=np.float64))

    def _save_components(self, f: "h5py.File", model: "Model") -> None:
        """Save named components to HDF5 file in CSR format."""
        components = model._state.components
        if not components:
            # Create empty components group
            comp_grp = f.create_group("components")
            comp_grp.create_dataset("names", data=[])
            comp_grp.create_dataset("types", data=[])
            comp_grp.create_dataset("offsets", data=np.array([0], dtype=np.int64))
            comp_grp.create_dataset("indices", data=np.array([], dtype=np.int64))
            return

        comp_grp = f.create_group("components")

        names = []
        types = []
        all_indices = []
        offsets = [0]

        for name, query in components.items():
            try:
                indices = query.evaluate(model.mesh, components)
                names.append(name)
                # Determine type from query class name
                query_type = type(query).__name__.lower()
                if "node" in query_type:
                    types.append("node")
                elif "element" in query_type:
                    types.append("element")
                elif "face" in query_type:
                    types.append("face")
                else:
                    types.append("node")  # Default

                all_indices.extend(indices)
                offsets.append(len(all_indices))
            except Exception:
                pass

        comp_grp.create_dataset("names", data=names)
        comp_grp.create_dataset("types", data=types)
        comp_grp.create_dataset("offsets", data=np.array(offsets, dtype=np.int64))
        comp_grp.create_dataset("indices", data=np.array(all_indices, dtype=np.int64))

    @classmethod
    def load(cls, path: str | Path) -> "Results":
        """Load results from HDF5 format.

        Opens the file for lazy access - data is loaded on demand.
        Remember to call close() when done, or use as context manager.

        Args:
            path: Path to .mops.h5 file.

        Returns:
            Results object with lazy-loaded data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is unsupported.

        Example::

            # Manual close
            results = Results.load("analysis.mops.h5")
            print(results.max_displacement())
            results.close()

            # Context manager (recommended)
            with Results.load("analysis.mops.h5") as results:
                disp = results.displacement()
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py package required for HDF5 save/load. "
                "Install with: pip install h5py"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        f = h5py.File(path, "r")

        # Validate format version
        version = f["/metadata/format_version"][()].decode()
        if version != FORMAT_VERSION:
            f.close()
            raise ValueError(f"Unsupported format version: {version}. Expected: {FORMAT_VERSION}")

        # Create Results with lazy loading
        results = cls(core_results=None, _hdf5_file=f, _lazy=True)

        # Load metadata
        results._mops_version = f["/metadata/mops_version"][()].decode()
        results._created_at = f["/metadata/created_at"][()].decode()
        results._analysis_type = f["/metadata/analysis_type"][()].decode()
        results._description = f["/metadata/description"][()].decode()

        # Load mesh metadata if available
        if "/mesh" in f:
            results._n_nodes = f["/mesh/nodes"].shape[0]
            results._n_elements = f["/mesh/elements"].shape[0]
            results._element_type = f["/mesh/element_type"][()].decode()

        return results

    def close(self) -> None:
        """Close the HDF5 file handle.

        Should be called when done with loaded results to free resources.
        No-op if results were not loaded from file.
        """
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def __enter__(self) -> "Results":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes file handle."""
        self.close()

    def __repr__(self) -> str:
        source = "file" if self._lazy else "solver"
        return (
            f"Results(source={source}, n_nodes={self.n_nodes}, n_elements={self.n_elements}, "
            f"max_disp={self.max_displacement():.3e}, max_vm={self.max_von_mises():.3e})"
        )


def _create_chunked_dataset(
    group: "h5py.Group",
    name: str,
    data: np.ndarray,
    chunks: tuple[int, ...],
) -> None:
    """Create a chunked, compressed HDF5 dataset.

    Args:
        group: HDF5 group to create dataset in.
        name: Dataset name.
        data: Numpy array data.
        chunks: Chunk shape (should not exceed data shape).
    """
    # Adjust chunk size to not exceed data shape
    actual_chunks = tuple(min(c, s) for c, s in zip(chunks, data.shape))

    group.create_dataset(
        name,
        data=data,
        chunks=actual_chunks,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
