"""Results class with HDF5 save/load functionality.

This module provides the Results class for storing and persisting FEA analysis results.
Results can be saved to HDF5 format for efficient storage and lazy loading of large datasets.

The module also provides query-optimized field accessors (NodeField, ElementField) that
allow selective data access using the Query DSL, enabling efficient partial reads from
HDF5 files.

Example::

    from mops import Model, solve, Nodes, Elements

    # Solve model
    results = solve(model)

    # Save to HDF5
    results.save("analysis.mops.h5", model=model, description="Cantilever analysis")

    # Load results (lazy loading - file stays open)
    loaded = Results.load("analysis.mops.h5")
    print(loaded.max_displacement())

    # Query-optimized access - only reads selected data from HDF5
    fixed_disp = loaded.displacement_field.where(Nodes.where(x=0))
    tip_stress = loaded.stress_field.where(Elements.touching(Nodes.where(x=100)))

    # Close when done
    loaded.close()

    # Or use context manager
    with Results.load("analysis.mops.h5") as results:
        disp = results.displacement()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

import mops

if TYPE_CHECKING:
    from mops._core import Results as CoreResults
    from mops.model import Model
    from mops.query import Query, NodeQuery, ElementQuery

# HDF5 format version
FORMAT_VERSION = "1.0"

# Type variable for field data
T = TypeVar("T", bound=np.ndarray)


class FieldAccessor(ABC, Generic[T]):
    """Base class for query-optimized field access.

    Field accessors provide lazy, query-aware access to result data. When
    backed by HDF5, they enable partial reads to minimize memory usage.

    The key methods are:
    - __getitem__(query): Get values for selected indices
    - where(query): Return a filtered field (lazy evaluation)
    - values(): Get full array (triggers load if lazy)
    """

    @abstractmethod
    def values(self) -> T:
        """Get the full field data as a numpy array."""
        ...

    @abstractmethod
    def __getitem__(self, query: "Query") -> np.ndarray:
        """Get field values for indices selected by query."""
        ...

    @abstractmethod
    def where(self, query: "Query") -> "FieldAccessor[T]":
        """Return a filtered field accessor (lazy evaluation)."""
        ...


class NodeField(FieldAccessor[NDArray[np.float64]]):
    """Query-optimized accessor for nodal field data (e.g., displacement).

    Provides efficient access to per-node result data using the Query DSL.
    When backed by HDF5, only reads the requested subset of data.

    Example::

        # Get displacement at specific nodes
        tip_disp = results.displacement_field[Nodes.where(x=100)]

        # Get filtered field (lazy - no data read yet)
        filtered = results.displacement_field.where(Nodes.where(x=0))
        data = filtered.values()  # Now reads from HDF5

        # Convenience methods
        mag = results.displacement_field.magnitude()  # Returns ScalarNodeField
    """

    def __init__(
        self,
        data: NDArray[np.float64] | None = None,
        *,
        h5file: object | None = None,
        h5path: str | None = None,
        mesh_h5path: str = "/mesh/nodes",
        indices: NDArray[np.int64] | None = None,
        parent: "NodeField | None" = None,
        components: dict[str, "Query"] | None = None,
    ) -> None:
        """Initialize NodeField.

        Args:
            data: Direct array data (for non-lazy access).
            h5file: HDF5 file handle for lazy loading.
            h5path: Path to dataset in HDF5 file.
            mesh_h5path: Path to mesh nodes in HDF5 (for query evaluation).
            indices: Pre-selected indices (for filtered fields).
            parent: Parent field (for filtered fields that inherit HDF5 refs).
            components: Named components for query resolution.
        """
        self._data = data
        self._h5file = h5file
        self._h5path = h5path
        self._mesh_h5path = mesh_h5path
        self._indices = indices
        self._parent = parent
        self._components = components or {}
        self._cache: NDArray[np.float64] | None = None

    def values(self) -> NDArray[np.float64]:
        """Get the full field data as a numpy array.

        For filtered fields, returns only the selected values.
        For HDF5-backed fields, triggers a read operation.

        Returns:
            Array of field values. Shape depends on whether filtered.
        """
        if self._cache is not None:
            return self._cache

        if self._data is not None:
            # In-memory data
            if self._indices is not None:
                self._cache = self._data[self._indices]
            else:
                self._cache = self._data
            return self._cache

        if self._parent is not None:
            # Filtered field - delegate to parent
            parent_data = self._parent.values()
            if self._indices is not None:
                self._cache = parent_data[self._indices]
            else:
                self._cache = parent_data
            return self._cache

        if self._h5file is not None and self._h5path is not None:
            # HDF5 lazy loading
            if self._indices is not None:
                # Optimized partial read - HDF5 fancy indexing
                # Sort indices for efficient read, then reorder
                sorted_idx = np.argsort(self._indices)
                sorted_indices = self._indices[sorted_idx]
                # Read sorted subset
                data = self._h5file[self._h5path][sorted_indices]
                # Restore original order
                inverse_idx = np.empty_like(sorted_idx)
                inverse_idx[sorted_idx] = np.arange(len(sorted_idx))
                self._cache = data[inverse_idx]
            else:
                self._cache = self._h5file[self._h5path][:]
            return self._cache

        raise RuntimeError("NodeField not initialized with data source")

    def __getitem__(self, query: "Query") -> np.ndarray:
        """Get field values for indices selected by query.

        Args:
            query: Node query to select indices.

        Returns:
            Array of values at selected nodes.
        """
        indices = self._evaluate_query(query)
        if self._indices is not None:
            # Already filtered - need to map indices
            # Find intersection of current indices with query result
            mask = np.isin(self._indices, indices)
            return self.values()[mask]
        return self._get_by_indices(indices)

    def _get_by_indices(self, indices: np.ndarray) -> np.ndarray:
        """Get values at specific indices, optimized for HDF5."""
        if len(indices) == 0:
            # Return empty array with correct shape
            full = self.values()
            return np.empty((0,) + full.shape[1:], dtype=full.dtype)

        if self._h5file is not None and self._h5path is not None and self._data is None:
            # Direct HDF5 read for subset
            sorted_idx = np.argsort(indices)
            sorted_indices = indices[sorted_idx]
            data = self._h5file[self._h5path][sorted_indices]
            inverse_idx = np.empty_like(sorted_idx)
            inverse_idx[sorted_idx] = np.arange(len(sorted_idx))
            return data[inverse_idx]

        # In-memory
        return self.values()[indices]

    def where(self, query: "Query") -> "NodeField":
        """Return a filtered NodeField (lazy evaluation).

        The query is evaluated against mesh data (from HDF5 or memory)
        to determine which indices to include. No result data is read
        until values() is called.

        Args:
            query: Node query to filter by.

        Returns:
            New NodeField that will return only selected values.

        Example::

            # Define a filter
            filtered = results.displacement_field.where(Nodes.where(x=0))

            # Data not read yet - just indices computed
            data = filtered.values()  # Now reads subset from HDF5
        """
        indices = self._evaluate_query(query)

        if self._indices is not None:
            # Already filtered - intersect with new selection
            mask = np.isin(self._indices, indices)
            new_indices = self._indices[mask]
        else:
            new_indices = indices

        return NodeField(
            data=self._data,
            h5file=self._h5file,
            h5path=self._h5path,
            mesh_h5path=self._mesh_h5path,
            indices=new_indices,
            parent=self if self._data is None else None,
            components=self._components,
        )

    def _evaluate_query(self, query: "Query") -> np.ndarray:
        """Evaluate a query to get node indices."""
        from mops.query import NodeQuery

        if not isinstance(query, NodeQuery):
            raise TypeError(
                f"NodeField requires NodeQuery, got {type(query).__name__}. "
                "Use Elements.* queries for ElementField."
            )

        # Build a minimal mesh-like object for query evaluation
        mesh = self._get_mesh_proxy()
        return query.evaluate(mesh, self._components)

    def _get_mesh_proxy(self) -> object:
        """Get mesh data for query evaluation."""
        if self._h5file is not None and self._mesh_h5path in self._h5file:
            # Create a proxy that provides coords from HDF5
            return _HDF5MeshProxy(self._h5file, self._mesh_h5path)

        raise RuntimeError(
            "Cannot evaluate query: no mesh data available. "
            "Results must be saved with model=model to enable queries."
        )

    def magnitude(self) -> "ScalarNodeField":
        """Compute magnitude of vector field (e.g., displacement magnitude).

        Returns:
            ScalarNodeField containing magnitude at each node.
        """
        data = self.values()
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("magnitude() requires vector field (shape Nx2 or Nx3)")
        mag = np.linalg.norm(data, axis=1)
        return ScalarNodeField(
            data=mag,
            indices=self._indices,
            components=self._components,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the field data."""
        if self._indices is not None:
            # Filtered
            full_shape = self._get_full_shape()
            return (len(self._indices),) + full_shape[1:]
        return self._get_full_shape()

    def _get_full_shape(self) -> tuple[int, ...]:
        """Get shape of unfiltered data."""
        if self._data is not None:
            return self._data.shape
        if self._h5file is not None and self._h5path is not None:
            return self._h5file[self._h5path].shape
        if self._cache is not None:
            return self._cache.shape
        raise RuntimeError("Cannot determine shape: no data source")

    def __repr__(self) -> str:
        filtered = " (filtered)" if self._indices is not None else ""
        lazy = " [lazy]" if self._h5file is not None and self._cache is None else ""
        return f"NodeField(shape={self.shape}{filtered}{lazy})"


class ScalarNodeField(FieldAccessor[NDArray[np.float64]]):
    """Scalar field on nodes (e.g., displacement magnitude, temperature).

    A simplified accessor for scalar per-node data. Supports aggregation
    methods like max(), min(), mean().
    """

    def __init__(
        self,
        data: NDArray[np.float64],
        *,
        indices: NDArray[np.int64] | None = None,
        components: dict[str, "Query"] | None = None,
    ) -> None:
        """Initialize ScalarNodeField.

        Args:
            data: 1D array of scalar values per node.
            indices: Pre-selected indices (for filtered fields).
            components: Named components for query resolution.
        """
        self._data = data
        self._indices = indices
        self._components = components or {}

    def values(self) -> NDArray[np.float64]:
        """Get the scalar values as a 1D array."""
        if self._indices is not None:
            return self._data[self._indices]
        return self._data

    def __getitem__(self, query: "Query") -> np.ndarray:
        """Get values for nodes selected by query."""
        from mops.query import NodeQuery

        if not isinstance(query, NodeQuery):
            raise TypeError(f"ScalarNodeField requires NodeQuery, got {type(query).__name__}")

        # For scalar fields derived from other fields, we work on in-memory data
        raise NotImplementedError(
            "Query-based access on derived scalar fields not yet supported. "
            "Use .values() to get the array."
        )

    def where(self, query: "Query") -> "ScalarNodeField":
        """Filter to specific nodes."""
        raise NotImplementedError(
            "where() on derived scalar fields not yet supported. "
            "Filter the parent field first."
        )

    def max(self) -> float:
        """Maximum value across all (selected) nodes."""
        return float(np.max(self.values()))

    def min(self) -> float:
        """Minimum value across all (selected) nodes."""
        return float(np.min(self.values()))

    def mean(self) -> float:
        """Mean value across all (selected) nodes."""
        return float(np.mean(self.values()))

    def argmax(self) -> int:
        """Index of maximum value."""
        return int(np.argmax(self.values()))

    def argmin(self) -> int:
        """Index of minimum value."""
        return int(np.argmin(self.values()))

    def __repr__(self) -> str:
        filtered = " (filtered)" if self._indices is not None else ""
        n = len(self._indices) if self._indices is not None else len(self._data)
        return f"ScalarNodeField(n={n}{filtered})"


class ElementField(FieldAccessor[NDArray[np.float64]]):
    """Query-optimized accessor for element field data (e.g., stress).

    Provides efficient access to per-element result data using the Query DSL.
    When backed by HDF5, only reads the requested subset of data.

    Example::

        # Get stress at specific elements
        tip_stress = results.stress_field[Elements.touching(Nodes.where(x=100))]

        # Get filtered field (lazy)
        filtered = results.stress_field.where(Elements.all())
        data = filtered.values()

        # Von Mises stress
        vm = results.von_mises_field.max()
    """

    def __init__(
        self,
        data: NDArray[np.float64] | None = None,
        *,
        h5file: object | None = None,
        h5path: str | None = None,
        mesh_nodes_h5path: str = "/mesh/nodes",
        mesh_elements_h5path: str = "/mesh/elements",
        indices: NDArray[np.int64] | None = None,
        parent: "ElementField | None" = None,
        components: dict[str, "Query"] | None = None,
    ) -> None:
        """Initialize ElementField.

        Args:
            data: Direct array data (for non-lazy access).
            h5file: HDF5 file handle for lazy loading.
            h5path: Path to dataset in HDF5 file.
            mesh_nodes_h5path: Path to mesh nodes in HDF5.
            mesh_elements_h5path: Path to mesh elements in HDF5.
            indices: Pre-selected indices (for filtered fields).
            parent: Parent field (for filtered fields).
            components: Named components for query resolution.
        """
        self._data = data
        self._h5file = h5file
        self._h5path = h5path
        self._mesh_nodes_h5path = mesh_nodes_h5path
        self._mesh_elements_h5path = mesh_elements_h5path
        self._indices = indices
        self._parent = parent
        self._components = components or {}
        self._cache: NDArray[np.float64] | None = None

    def values(self) -> NDArray[np.float64]:
        """Get the full field data as a numpy array."""
        if self._cache is not None:
            return self._cache

        if self._data is not None:
            if self._indices is not None:
                self._cache = self._data[self._indices]
            else:
                self._cache = self._data
            return self._cache

        if self._parent is not None:
            parent_data = self._parent.values()
            if self._indices is not None:
                self._cache = parent_data[self._indices]
            else:
                self._cache = parent_data
            return self._cache

        if self._h5file is not None and self._h5path is not None:
            if self._indices is not None:
                sorted_idx = np.argsort(self._indices)
                sorted_indices = self._indices[sorted_idx]
                data = self._h5file[self._h5path][sorted_indices]
                inverse_idx = np.empty_like(sorted_idx)
                inverse_idx[sorted_idx] = np.arange(len(sorted_idx))
                self._cache = data[inverse_idx]
            else:
                self._cache = self._h5file[self._h5path][:]
            return self._cache

        raise RuntimeError("ElementField not initialized with data source")

    def __getitem__(self, query: "Query") -> np.ndarray:
        """Get field values for indices selected by query."""
        indices = self._evaluate_query(query)
        if self._indices is not None:
            mask = np.isin(self._indices, indices)
            return self.values()[mask]
        return self._get_by_indices(indices)

    def _get_by_indices(self, indices: np.ndarray) -> np.ndarray:
        """Get values at specific indices, optimized for HDF5."""
        if len(indices) == 0:
            full = self.values()
            return np.empty((0,) + full.shape[1:], dtype=full.dtype)

        if self._h5file is not None and self._h5path is not None and self._data is None:
            sorted_idx = np.argsort(indices)
            sorted_indices = indices[sorted_idx]
            data = self._h5file[self._h5path][sorted_indices]
            inverse_idx = np.empty_like(sorted_idx)
            inverse_idx[sorted_idx] = np.arange(len(sorted_idx))
            return data[inverse_idx]

        return self.values()[indices]

    def where(self, query: "Query") -> "ElementField":
        """Return a filtered ElementField (lazy evaluation)."""
        indices = self._evaluate_query(query)

        if self._indices is not None:
            mask = np.isin(self._indices, indices)
            new_indices = self._indices[mask]
        else:
            new_indices = indices

        return ElementField(
            data=self._data,
            h5file=self._h5file,
            h5path=self._h5path,
            mesh_nodes_h5path=self._mesh_nodes_h5path,
            mesh_elements_h5path=self._mesh_elements_h5path,
            indices=new_indices,
            parent=self if self._data is None else None,
            components=self._components,
        )

    def _evaluate_query(self, query: "Query") -> np.ndarray:
        """Evaluate a query to get element indices."""
        from mops.query import ElementQuery

        if not isinstance(query, ElementQuery):
            raise TypeError(
                f"ElementField requires ElementQuery, got {type(query).__name__}. "
                "Use Nodes.* queries for NodeField."
            )

        mesh = self._get_mesh_proxy()
        return query.evaluate(mesh, self._components)

    def _get_mesh_proxy(self) -> object:
        """Get mesh data for query evaluation."""
        if self._h5file is not None:
            if self._mesh_nodes_h5path in self._h5file:
                return _HDF5MeshProxy(
                    self._h5file,
                    self._mesh_nodes_h5path,
                    self._mesh_elements_h5path,
                )

        raise RuntimeError(
            "Cannot evaluate query: no mesh data available. "
            "Results must be saved with model=model to enable queries."
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the field data."""
        if self._indices is not None:
            full_shape = self._get_full_shape()
            return (len(self._indices),) + full_shape[1:]
        return self._get_full_shape()

    def _get_full_shape(self) -> tuple[int, ...]:
        """Get shape of unfiltered data."""
        if self._data is not None:
            return self._data.shape
        if self._h5file is not None and self._h5path is not None:
            return self._h5file[self._h5path].shape
        if self._cache is not None:
            return self._cache.shape
        raise RuntimeError("Cannot determine shape: no data source")

    def max(self) -> float:
        """Maximum value (for scalar fields) or max norm (for vector/tensor)."""
        data = self.values()
        if data.ndim == 1:
            return float(np.max(data))
        return float(np.max(np.linalg.norm(data, axis=1)))

    def min(self) -> float:
        """Minimum value (for scalar fields) or min norm (for vector/tensor)."""
        data = self.values()
        if data.ndim == 1:
            return float(np.min(data))
        return float(np.min(np.linalg.norm(data, axis=1)))

    def mean(self) -> float:
        """Mean value (for scalar fields) or mean norm (for vector/tensor)."""
        data = self.values()
        if data.ndim == 1:
            return float(np.mean(data))
        return float(np.mean(np.linalg.norm(data, axis=1)))

    def __repr__(self) -> str:
        filtered = " (filtered)" if self._indices is not None else ""
        lazy = " [lazy]" if self._h5file is not None and self._cache is None else ""
        return f"ElementField(shape={self.shape}{filtered}{lazy})"


class ScalarElementField(FieldAccessor[NDArray[np.float64]]):
    """Scalar field on elements (e.g., von Mises stress).

    A simplified accessor for scalar per-element data with aggregation methods.
    """

    def __init__(
        self,
        data: NDArray[np.float64] | None = None,
        *,
        h5file: object | None = None,
        h5path: str | None = None,
        mesh_nodes_h5path: str = "/mesh/nodes",
        mesh_elements_h5path: str = "/mesh/elements",
        indices: NDArray[np.int64] | None = None,
        components: dict[str, "Query"] | None = None,
    ) -> None:
        """Initialize ScalarElementField."""
        self._data = data
        self._h5file = h5file
        self._h5path = h5path
        self._mesh_nodes_h5path = mesh_nodes_h5path
        self._mesh_elements_h5path = mesh_elements_h5path
        self._indices = indices
        self._components = components or {}
        self._cache: NDArray[np.float64] | None = None

    def values(self) -> NDArray[np.float64]:
        """Get the scalar values as a 1D array."""
        if self._cache is not None:
            return self._cache

        if self._data is not None:
            if self._indices is not None:
                self._cache = self._data[self._indices]
            else:
                self._cache = self._data
            return self._cache

        if self._h5file is not None and self._h5path is not None:
            if self._indices is not None:
                sorted_idx = np.argsort(self._indices)
                sorted_indices = self._indices[sorted_idx]
                data = self._h5file[self._h5path][sorted_indices]
                inverse_idx = np.empty_like(sorted_idx)
                inverse_idx[sorted_idx] = np.arange(len(sorted_idx))
                self._cache = data[inverse_idx]
            else:
                self._cache = self._h5file[self._h5path][:]
            return self._cache

        raise RuntimeError("ScalarElementField not initialized with data source")

    def __getitem__(self, query: "Query") -> np.ndarray:
        """Get values for elements selected by query."""
        indices = self._evaluate_query(query)
        if self._indices is not None:
            mask = np.isin(self._indices, indices)
            return self.values()[mask]
        return self._get_by_indices(indices)

    def _get_by_indices(self, indices: np.ndarray) -> np.ndarray:
        """Get values at specific indices."""
        if len(indices) == 0:
            return np.empty(0, dtype=np.float64)

        if self._h5file is not None and self._h5path is not None and self._data is None:
            sorted_idx = np.argsort(indices)
            sorted_indices = indices[sorted_idx]
            data = self._h5file[self._h5path][sorted_indices]
            inverse_idx = np.empty_like(sorted_idx)
            inverse_idx[sorted_idx] = np.arange(len(sorted_idx))
            return data[inverse_idx]

        return self.values()[indices]

    def _evaluate_query(self, query: "Query") -> np.ndarray:
        """Evaluate a query to get element indices."""
        from mops.query import ElementQuery

        if not isinstance(query, ElementQuery):
            raise TypeError(
                f"ScalarElementField requires ElementQuery, got {type(query).__name__}"
            )

        mesh = self._get_mesh_proxy()
        return query.evaluate(mesh, self._components)

    def _get_mesh_proxy(self) -> object:
        """Get mesh data for query evaluation."""
        if self._h5file is not None:
            if self._mesh_nodes_h5path in self._h5file:
                return _HDF5MeshProxy(
                    self._h5file,
                    self._mesh_nodes_h5path,
                    self._mesh_elements_h5path,
                )

        raise RuntimeError(
            "Cannot evaluate query: no mesh data available. "
            "Results must be saved with model=model to enable queries."
        )

    def where(self, query: "Query") -> "ScalarElementField":
        """Return a filtered ScalarElementField."""
        indices = self._evaluate_query(query)

        if self._indices is not None:
            mask = np.isin(self._indices, indices)
            new_indices = self._indices[mask]
        else:
            new_indices = indices

        return ScalarElementField(
            data=self._data,
            h5file=self._h5file,
            h5path=self._h5path,
            mesh_nodes_h5path=self._mesh_nodes_h5path,
            mesh_elements_h5path=self._mesh_elements_h5path,
            indices=new_indices,
            components=self._components,
        )

    def max(self) -> float:
        """Maximum value across all (selected) elements."""
        return float(np.max(self.values()))

    def min(self) -> float:
        """Minimum value across all (selected) elements."""
        return float(np.min(self.values()))

    def mean(self) -> float:
        """Mean value across all (selected) elements."""
        return float(np.mean(self.values()))

    def argmax(self) -> int:
        """Index of maximum value."""
        return int(np.argmax(self.values()))

    def argmin(self) -> int:
        """Index of minimum value."""
        return int(np.argmin(self.values()))

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the field data."""
        if self._indices is not None:
            return (len(self._indices),)
        return self._get_full_shape()

    def _get_full_shape(self) -> tuple[int, ...]:
        """Get shape of unfiltered data."""
        if self._data is not None:
            return self._data.shape
        if self._h5file is not None and self._h5path is not None:
            return self._h5file[self._h5path].shape
        if self._cache is not None:
            return self._cache.shape
        raise RuntimeError("Cannot determine shape: no data source")

    def __repr__(self) -> str:
        filtered = " (filtered)" if self._indices is not None else ""
        lazy = " [lazy]" if self._h5file is not None and self._cache is None else ""
        n = len(self._indices) if self._indices is not None else self.shape[0]
        return f"ScalarElementField(n={n}{filtered}{lazy})"


class _HDF5MeshProxy:
    """Minimal mesh-like proxy for query evaluation against HDF5 data.

    Provides the interface expected by Query.evaluate() using HDF5 datasets.
    """

    def __init__(
        self,
        h5file: object,
        nodes_path: str,
        elements_path: str | None = None,
    ) -> None:
        """Initialize proxy.

        Args:
            h5file: Open h5py.File handle.
            nodes_path: HDF5 path to nodes dataset.
            elements_path: HDF5 path to elements dataset (optional).
        """
        self._h5file = h5file
        self._nodes_path = nodes_path
        self._elements_path = elements_path
        self._coords_cache: np.ndarray | None = None
        self._elements_cache: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self._h5file[self._nodes_path].shape[0]

    @property
    def n_elements(self) -> int:
        """Number of elements."""
        if self._elements_path is None:
            raise RuntimeError("Elements path not set")
        return self._h5file[self._elements_path].shape[0]

    @property
    def coords(self) -> np.ndarray:
        """Node coordinates (loaded on first access)."""
        if self._coords_cache is None:
            self._coords_cache = self._h5file[self._nodes_path][:]
        return self._coords_cache

    @property
    def elements(self) -> np.ndarray:
        """Element connectivity (loaded on first access)."""
        if self._elements_path is None:
            raise RuntimeError("Elements path not set")
        if self._elements_cache is None:
            self._elements_cache = self._h5file[self._elements_path][:]
        return self._elements_cache


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

    # =========================================================================
    # Query-Optimized Field Accessors
    # =========================================================================

    @property
    def displacement_field(self) -> NodeField:
        """Query-optimized accessor for displacement field.

        Returns a NodeField that supports the Query DSL for efficient
        partial access to displacement data.

        Example::

            # Get displacement at fixed boundary
            fixed_disp = results.displacement_field.where(Nodes.where(x=0))

            # Direct indexing with query
            tip_disp = results.displacement_field[Nodes.where(x=100)]

            # Magnitude
            mag = results.displacement_field.magnitude()
        """
        if self._lazy and self._h5file is not None:
            # HDF5 lazy loading mode
            return NodeField(
                h5file=self._h5file,
                h5path="/solution/displacement",
                mesh_h5path="/mesh/nodes",
            )

        if self._core is not None:
            # In-memory mode
            return NodeField(data=self.displacement())

        raise RuntimeError("Results not initialized")

    @property
    def stress_field(self) -> ElementField:
        """Query-optimized accessor for stress tensor field.

        Returns an ElementField that supports the Query DSL for efficient
        partial access to stress data.

        Example::

            # Get stress at tip elements
            tip_stress = results.stress_field.where(
                Elements.touching(Nodes.where(x=100))
            )

            # Direct indexing with query
            fixed_stress = results.stress_field[Elements.attached_to(Nodes.where(x=0))]
        """
        if self._lazy and self._h5file is not None:
            return ElementField(
                h5file=self._h5file,
                h5path="/stress/element",
                mesh_nodes_h5path="/mesh/nodes",
                mesh_elements_h5path="/mesh/elements",
            )

        if self._core is not None:
            return ElementField(data=self.stress())

        raise RuntimeError("Results not initialized")

    @property
    def von_mises_field(self) -> ScalarElementField:
        """Query-optimized accessor for von Mises stress field.

        Returns a ScalarElementField that supports the Query DSL for efficient
        partial access and aggregation operations.

        Example::

            # Max von Mises at tip
            tip_vm = results.von_mises_field.where(
                Elements.touching(Nodes.where(x=100))
            ).max()

            # Direct indexing
            fixed_vm = results.von_mises_field[Elements.attached_to(Nodes.where(x=0))]
        """
        if self._lazy and self._h5file is not None:
            return ScalarElementField(
                h5file=self._h5file,
                h5path="/stress/element_von_mises",
                mesh_nodes_h5path="/mesh/nodes",
                mesh_elements_h5path="/mesh/elements",
            )

        if self._core is not None:
            return ScalarElementField(data=self.von_mises())

        raise RuntimeError("Results not initialized")

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

    # =========================================================================
    # Derived Quantities
    # =========================================================================

    def principal_stresses(self) -> NDArray[np.float64]:
        """Get principal stresses per element.

        Computes the eigenvalues of the stress tensor for each element.
        Principal stresses are returned in descending order: σ₁ ≥ σ₂ ≥ σ₃.

        Returns:
            (n_elements, 3) array of principal stresses [σ₁, σ₂, σ₃] per element.

        Note:
            This is computed on-demand from the stress tensor data. For HDF5-backed
            results, the computation is performed once and cached in memory.
        """
        if not hasattr(self, "_principal_stress_cache"):
            self._principal_stress_cache = None

        if self._principal_stress_cache is not None:
            return self._principal_stress_cache

        # Check HDF5 cache
        if self._lazy and self._h5file is not None:
            if "/stress/element_principal" in self._h5file:
                self._principal_stress_cache = self._h5file["/stress/element_principal"][:]
                return self._principal_stress_cache

        # Compute from stress tensor
        from mops.derived import principal_stresses as compute_principal

        self._principal_stress_cache = compute_principal(self.stress())
        return self._principal_stress_cache

    def tresca(self) -> NDArray[np.float64]:
        """Get Tresca (maximum shear) equivalent stress per element.

        The Tresca stress is defined as: σ_tresca = σ₁ - σ₃

        This criterion is used in maximum shear stress failure theory
        and is more conservative than von Mises for many materials.

        Returns:
            (n_elements,) array of Tresca stress values.
        """
        if not hasattr(self, "_tresca_cache"):
            self._tresca_cache = None

        if self._tresca_cache is not None:
            return self._tresca_cache

        # Check HDF5 cache
        if self._lazy and self._h5file is not None:
            if "/stress/element_tresca" in self._h5file:
                self._tresca_cache = self._h5file["/stress/element_tresca"][:]
                return self._tresca_cache

        # Compute from stress tensor
        from mops.derived import tresca_stress

        self._tresca_cache = tresca_stress(self.stress())
        return self._tresca_cache

    def max_tresca(self) -> float:
        """Get maximum Tresca stress.

        Returns:
            Maximum Tresca stress across all elements.
        """
        return float(np.max(self.tresca()))

    def max_shear_stress(self) -> NDArray[np.float64]:
        """Get maximum shear stress per element.

        τ_max = (σ₁ - σ₃) / 2

        Returns:
            (n_elements,) array of maximum shear stress values.
        """
        return self.tresca() / 2.0

    def hydrostatic_stress(self) -> NDArray[np.float64]:
        """Get hydrostatic (mean) stress per element.

        σ_h = (σ_xx + σ_yy + σ_zz) / 3

        Returns:
            (n_elements,) array of hydrostatic stress values.
        """
        if not hasattr(self, "_hydrostatic_cache"):
            self._hydrostatic_cache = None

        if self._hydrostatic_cache is not None:
            return self._hydrostatic_cache

        # Check HDF5 cache
        if self._lazy and self._h5file is not None:
            if "/stress/element_hydrostatic" in self._h5file:
                self._hydrostatic_cache = self._h5file["/stress/element_hydrostatic"][:]
                return self._hydrostatic_cache

        # Compute from stress tensor
        from mops.derived import hydrostatic_stress

        self._hydrostatic_cache = hydrostatic_stress(self.stress())
        return self._hydrostatic_cache

    def pressure(self) -> NDArray[np.float64]:
        """Get pressure (negative hydrostatic stress) per element.

        p = -σ_h = -(σ_xx + σ_yy + σ_zz) / 3

        Returns:
            (n_elements,) array of pressure values.
        """
        return -self.hydrostatic_stress()

    def stress_intensity(self) -> NDArray[np.float64]:
        """Get stress intensity per element.

        σ_int = max(|σ₁ - σ₂|, |σ₂ - σ₃|, |σ₃ - σ₁|)

        This is used in some codes as an alternative to Tresca stress.

        Returns:
            (n_elements,) array of stress intensity values.
        """
        if not hasattr(self, "_intensity_cache"):
            self._intensity_cache = None

        if self._intensity_cache is not None:
            return self._intensity_cache

        # Check HDF5 cache
        if self._lazy and self._h5file is not None:
            if "/stress/element_intensity" in self._h5file:
                self._intensity_cache = self._h5file["/stress/element_intensity"][:]
                return self._intensity_cache

        # Compute from principal stresses
        principals = self.principal_stresses()
        s1, s2, s3 = principals[:, 0], principals[:, 1], principals[:, 2]
        self._intensity_cache = np.maximum(
            np.abs(s1 - s2), np.maximum(np.abs(s2 - s3), np.abs(s3 - s1))
        )
        return self._intensity_cache

    @property
    def principal_stress_field(self) -> ScalarElementField:
        """Query-optimized accessor for maximum principal stress field (σ₁).

        Returns a ScalarElementField for the first (maximum) principal stress.

        Example::

            # Max principal stress at tip
            max_s1 = results.principal_stress_field.max()
        """
        principals = self.principal_stresses()
        return ScalarElementField(
            data=principals[:, 0],
            h5file=self._h5file if self._lazy else None,
            h5path="/stress/element_principal",  # Would need HDF5 indexing
            mesh_nodes_h5path="/mesh/nodes",
            mesh_elements_h5path="/mesh/elements",
        )

    @property
    def tresca_field(self) -> ScalarElementField:
        """Query-optimized accessor for Tresca stress field.

        Example::

            # Max Tresca stress
            max_tresca = results.tresca_field.max()
        """
        if self._lazy and self._h5file is not None:
            return ScalarElementField(
                h5file=self._h5file,
                h5path="/stress/element_tresca",
                mesh_nodes_h5path="/mesh/nodes",
                mesh_elements_h5path="/mesh/elements",
            )

        return ScalarElementField(data=self.tresca())

    @property
    def hydrostatic_field(self) -> ScalarElementField:
        """Query-optimized accessor for hydrostatic stress field.

        Example::

            # Mean hydrostatic stress
            avg_hydro = results.hydrostatic_field.mean()
        """
        if self._lazy and self._h5file is not None:
            return ScalarElementField(
                h5file=self._h5file,
                h5path="/stress/element_hydrostatic",
                mesh_nodes_h5path="/mesh/nodes",
                mesh_elements_h5path="/mesh/elements",
            )

        return ScalarElementField(data=self.hydrostatic_stress())

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

            # Derived quantities - computed on-demand and cached in HDF5
            # Principal stresses
            principal_data = self.principal_stresses()
            _create_chunked_dataset(
                stress_grp, "element_principal", principal_data,
                chunks=(min(1000, len(principal_data)), 3),
            )

            # Tresca stress
            tresca_data = self.tresca()
            _create_chunked_dataset(
                stress_grp, "element_tresca", tresca_data,
                chunks=(min(10000, len(tresca_data)),),
            )

            # Hydrostatic stress
            hydro_data = self.hydrostatic_stress()
            _create_chunked_dataset(
                stress_grp, "element_hydrostatic", hydro_data,
                chunks=(min(10000, len(hydro_data)),),
            )

            # Stress intensity
            intensity_data = self.stress_intensity()
            _create_chunked_dataset(
                stress_grp, "element_intensity", intensity_data,
                chunks=(min(10000, len(intensity_data)),),
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

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot(
        self,
        field: str = "von_mises",
        *,
        model: "Model | None" = None,
        cmap: str | None = None,
        show_edges: bool = True,
        show_scalar_bar: bool = True,
        scalar_bar_title: str | None = None,
        clim: tuple[float, float] | None = None,
        deformed: bool = False,
        scale_factor: float = 1.0,
        opacity: float = 1.0,
        window_size: tuple[int, int] = (1024, 768),
        background: str = "white",
        title: str | None = None,
        screenshot: str | None = None,
        notebook: bool | None = None,
        off_screen: bool | None = None,
        return_plotter: bool = False,
    ) -> object | None:
        """Plot results using PyVista.

        Creates an interactive 3D visualization of the analysis results.
        In Jupyter notebooks, renders inline. For scripts, opens an interactive window.

        Args:
            field: Field to visualize. Options:
                - "displacement", "displacement_magnitude", "displacement_x/y/z"
                - "von_mises", "stress_xx/yy/zz/xy/yz/xz"
                - "principal_1/2/3", "tresca", "hydrostatic"
            model: Model object (required for in-memory results, optional if
                results were loaded from HDF5 with mesh data).
            cmap: Colormap name (e.g., "jet", "viridis"). Auto-selected if None.
            show_edges: Show mesh edges (default True).
            show_scalar_bar: Show color bar (default True).
            scalar_bar_title: Custom title for scalar bar.
            clim: Color limits as (min, max). Auto-scaled if None.
            deformed: Show deformed shape (default False).
            scale_factor: Deformation scale factor (default 1.0).
            opacity: Surface opacity 0-1 (default 1.0).
            window_size: Window size in pixels (default (1024, 768)).
            background: Background color (default "white").
            title: Plot title.
            screenshot: Save screenshot to this path.
            notebook: Force notebook mode (auto-detected if None).
            off_screen: Render off-screen without display.
            return_plotter: Return PyVista plotter instead of showing.

        Returns:
            PyVista Plotter if return_plotter=True, else None.

        Raises:
            ImportError: If pyvista is not installed.
            ValueError: If mesh data not available and model not provided.

        Example::

            # Basic usage
            results.plot("von_mises")

            # With deformed shape
            results.plot("displacement_magnitude", deformed=True, scale_factor=100)

            # Save to file
            results.plot("von_mises", screenshot="stress.png")
        """
        from mops.viz import plot_results

        return plot_results(
            self,
            model=model,
            field=field,
            cmap=cmap,
            show_edges=show_edges,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_title=scalar_bar_title,
            clim=clim,
            deformed=deformed,
            scale_factor=scale_factor,
            opacity=opacity,
            window_size=window_size,
            background=background,
            title=title,
            screenshot=screenshot,
            notebook=notebook,
            off_screen=off_screen,
            return_plotter=return_plotter,
        )

    def export(
        self,
        path: str,
        *,
        model: "Model | None" = None,
        deformed: bool = False,
        scale_factor: float = 1.0,
        fields: list[str] | None = None,
    ) -> None:
        """Export results to VTU format for ParaView.

        Creates a VTK Unstructured Grid file (.vtu) containing mesh geometry
        and result fields. Can be opened in ParaView or other VTK-compatible tools.

        Args:
            path: Output file path (should end in .vtu).
            model: Model object (required for in-memory results, optional if
                results were loaded from HDF5 with mesh data).
            deformed: Export deformed shape (default False).
            scale_factor: Deformation scale factor (default 1.0).
            fields: List of fields to include. If None, exports all available fields:
                displacement, displacement_magnitude, von_mises, principal stresses,
                tresca, hydrostatic.

        Raises:
            ImportError: If pyvista is not installed.
            ValueError: If mesh data not available and model not provided.

        Example::

            # Export all fields
            results.export("output.vtu", model=model)

            # Export specific fields
            results.export("output.vtu", model=model, fields=["von_mises", "displacement"])

            # Export deformed shape
            results.export("deformed.vtu", model=model, deformed=True, scale_factor=100)
        """
        from mops.viz import export_vtu

        export_vtu(
            self,
            path,
            model=model,
            deformed=deformed,
            scale_factor=scale_factor,
            fields=fields,
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
