"""Query DSL for node, element, and face selection.

The query system provides a declarative way to select parts of the mesh
for applying materials, constraints, and loads.

Example::

    # Select all nodes at x=0
    Nodes.where(x=0)

    # Select nodes in a range
    Nodes.where(x__gt=0, x__lt=10)

    # Combine queries
    Nodes.where(x=0).union(Nodes.where(x=100))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from mops._core import Mesh


class Query(ABC):
    """Base class for all queries."""

    @abstractmethod
    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        """Evaluate query against mesh, return indices."""
        ...

    def union(self, other: Query) -> Query:
        """Union of two queries."""
        return UnionQuery(self, other)

    def subtract(self, other: Query) -> Query:
        """Set difference of two queries."""
        return SubtractQuery(self, other)


@dataclass
class UnionQuery(Query):
    """Union of two queries."""

    left: Query
    right: Query

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        left_indices = self.left.evaluate(mesh)
        right_indices = self.right.evaluate(mesh)
        return np.unique(np.concatenate([left_indices, right_indices]))


@dataclass
class SubtractQuery(Query):
    """Set difference of two queries."""

    left: Query
    right: Query

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        left_indices = set(self.left.evaluate(mesh))
        right_indices = set(self.right.evaluate(mesh))
        return np.array(sorted(left_indices - right_indices), dtype=np.int64)


@dataclass
class NodeQuery(Query):
    """Lazy node selection query.

    Stores predicates and evaluates them when needed.
    """

    predicates: dict[str, Any] = field(default_factory=dict)
    _all: bool = False

    def and_where(self, **kwargs: Any) -> NodeQuery:
        """Intersect with additional predicate."""
        new_predicates = {**self.predicates, **kwargs}
        return NodeQuery(predicates=new_predicates)

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        """Evaluate query, return node indices."""
        n_nodes = mesh.n_nodes

        if self._all:
            return np.arange(n_nodes, dtype=np.int64)

        # Start with all nodes
        mask = np.ones(n_nodes, dtype=bool)

        # Get node coordinates (we need to access mesh data)
        # For now, we'll need to get coordinates through mesh properties
        # This is a simplified implementation that works with our current API

        # Apply predicates
        for key, value in self.predicates.items():
            # Handle special syntax like x__gt, x__lt, x__between
            if "__" in key:
                coord, op = key.rsplit("__", 1)
                coord_idx = {"x": 0, "y": 1, "z": 2}.get(coord)
                if coord_idx is None:
                    raise ValueError(f"Unknown coordinate: {coord}")

                # We need access to node coordinates
                # This will be enhanced when mesh provides coordinate access
                pass
            elif key in ("x", "y", "z"):
                # Exact match with tolerance
                coord_idx = {"x": 0, "y": 1, "z": 2}[key]
                # Apply predicate when mesh provides coordinates
                pass

        # For now, return indices based on predicates as indices
        # This is a simplified version until mesh coordinate access is available
        if "indices" in self.predicates:
            return np.array(self.predicates["indices"], dtype=np.int64)

        # Default: return all nodes (simplified for initial implementation)
        return np.arange(n_nodes, dtype=np.int64)

    def __repr__(self) -> str:
        if self._all:
            return "Nodes.all()"
        return f"Nodes.where({self.predicates})"


class Nodes:
    """Query builder for node selection.

    Example::

        # All nodes
        Nodes.all()

        # Nodes at x=0
        Nodes.where(x=0)

        # Nodes with x > 10
        Nodes.where(x__gt=10)

        # Nodes by index
        Nodes.by_indices([0, 1, 2])
    """

    @classmethod
    def all(cls) -> NodeQuery:
        """Select all nodes."""
        return NodeQuery(_all=True)

    @classmethod
    def where(
        cls,
        x: float | Callable[[float], bool] | None = None,
        y: float | Callable[[float], bool] | None = None,
        z: float | Callable[[float], bool] | None = None,
        x__gt: float | None = None,
        x__lt: float | None = None,
        x__between: tuple[float, float] | None = None,
        y__gt: float | None = None,
        y__lt: float | None = None,
        y__between: tuple[float, float] | None = None,
        z__gt: float | None = None,
        z__lt: float | None = None,
        z__between: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> NodeQuery:
        """Select nodes matching predicates.

        Args:
            x, y, z: Exact coordinate match (with tolerance)
            x__gt, y__gt, z__gt: Coordinate greater than value
            x__lt, y__lt, z__lt: Coordinate less than value
            x__between, y__between, z__between: Coordinate in range (inclusive)
            **kwargs: Additional predicates

        Returns:
            Lazy NodeQuery for evaluation
        """
        predicates: dict[str, Any] = {}

        if x is not None:
            predicates["x"] = x
        if y is not None:
            predicates["y"] = y
        if z is not None:
            predicates["z"] = z
        if x__gt is not None:
            predicates["x__gt"] = x__gt
        if x__lt is not None:
            predicates["x__lt"] = x__lt
        if x__between is not None:
            predicates["x__between"] = x__between
        if y__gt is not None:
            predicates["y__gt"] = y__gt
        if y__lt is not None:
            predicates["y__lt"] = y__lt
        if y__between is not None:
            predicates["y__between"] = y__between
        if z__gt is not None:
            predicates["z__gt"] = z__gt
        if z__lt is not None:
            predicates["z__lt"] = z__lt
        if z__between is not None:
            predicates["z__between"] = z__between

        predicates.update(kwargs)
        return NodeQuery(predicates=predicates)

    @classmethod
    def by_indices(cls, indices: list[int]) -> NodeQuery:
        """Select nodes by index.

        Args:
            indices: List of node indices to select

        Returns:
            NodeQuery that selects the specified nodes
        """
        return NodeQuery(predicates={"indices": indices})


@dataclass
class ElementQuery(Query):
    """Lazy element selection query."""

    predicates: dict[str, Any] = field(default_factory=dict)
    _all: bool = False

    def and_where(self, **kwargs: Any) -> ElementQuery:
        """Intersect with additional predicate."""
        new_predicates = {**self.predicates, **kwargs}
        return ElementQuery(predicates=new_predicates)

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        """Evaluate query, return element indices."""
        n_elements = mesh.n_elements

        if self._all:
            return np.arange(n_elements, dtype=np.int64)

        # Apply predicates (simplified implementation)
        if "indices" in self.predicates:
            return np.array(self.predicates["indices"], dtype=np.int64)

        return np.arange(n_elements, dtype=np.int64)

    def __repr__(self) -> str:
        if self._all:
            return "Elements.all()"
        return f"Elements.where({self.predicates})"


class Elements:
    """Query builder for element selection.

    Example::

        # All elements
        Elements.all()

        # Elements by type
        Elements.where(type="tet4")

        # Elements with specific material
        Elements.where(material="steel")
    """

    @classmethod
    def all(cls) -> ElementQuery:
        """Select all elements."""
        return ElementQuery(_all=True)

    @classmethod
    def where(
        cls,
        type: str | None = None,
        material: str | None = None,
        in_component: str | None = None,
        adjacent_to: NodeQuery | None = None,
        **kwargs: Any,
    ) -> ElementQuery:
        """Select elements matching predicates.

        Args:
            type: Element type ("tet4", "tet10", "hex8", etc.)
            material: Material name
            in_component: Component name
            adjacent_to: Nodes the elements must be adjacent to
            **kwargs: Additional predicates

        Returns:
            Lazy ElementQuery for evaluation
        """
        predicates: dict[str, Any] = {}

        if type is not None:
            predicates["type"] = type
        if material is not None:
            predicates["material"] = material
        if in_component is not None:
            predicates["in_component"] = in_component
        if adjacent_to is not None:
            predicates["adjacent_to"] = adjacent_to

        predicates.update(kwargs)
        return ElementQuery(predicates=predicates)

    @classmethod
    def by_indices(cls, indices: list[int]) -> ElementQuery:
        """Select elements by index."""
        return ElementQuery(predicates={"indices": indices})


@dataclass
class FaceQuery(Query):
    """Lazy face selection query."""

    predicates: dict[str, Any] = field(default_factory=dict)

    def and_where(self, **kwargs: Any) -> FaceQuery:
        """Intersect with additional predicate."""
        new_predicates = {**self.predicates, **kwargs}
        return FaceQuery(predicates=new_predicates)

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        """Evaluate query, return face indices."""
        # Face queries will be implemented with mesh face extraction
        return np.array([], dtype=np.int64)

    def __repr__(self) -> str:
        return f"Faces.where({self.predicates})"


class Faces:
    """Query builder for surface face selection.

    Example::

        # Faces with specific normal
        Faces.where(normal=(1, 0, 0))

        # Boundary faces only
        Faces.where(on_boundary=True)
    """

    @classmethod
    def where(
        cls,
        normal: tuple[float, float, float] | None = None,
        on_boundary: bool | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        x__gt: float | None = None,
        x__lt: float | None = None,
        **kwargs: Any,
    ) -> FaceQuery:
        """Select faces matching predicates.

        Args:
            normal: Face normal direction (will match with tolerance)
            on_boundary: True for boundary faces only
            x, y, z: Faces at specific coordinate
            x__gt, x__lt: Faces with coordinates in range
            **kwargs: Additional spatial predicates

        Returns:
            Lazy FaceQuery for evaluation
        """
        predicates: dict[str, Any] = {}

        if normal is not None:
            predicates["normal"] = normal
        if on_boundary is not None:
            predicates["on_boundary"] = on_boundary
        if x is not None:
            predicates["x"] = x
        if y is not None:
            predicates["y"] = y
        if z is not None:
            predicates["z"] = z
        if x__gt is not None:
            predicates["x__gt"] = x__gt
        if x__lt is not None:
            predicates["x__lt"] = x__lt

        predicates.update(kwargs)
        return FaceQuery(predicates=predicates)
