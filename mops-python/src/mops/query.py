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

        # Handle index-based selection first
        if "indices" in self.predicates:
            return np.array(self.predicates["indices"], dtype=np.int64)

        # Get node coordinates for spatial predicates
        coords = mesh.coords  # Nx3 numpy array
        if coords.shape[0] == 0:
            return np.array([], dtype=np.int64)

        # Start with all nodes selected
        mask = np.ones(n_nodes, dtype=bool)

        # Default tolerance for coordinate comparisons
        tol = 1e-10

        # Apply predicates
        for key, value in self.predicates.items():
            if key == "indices":
                continue  # Already handled above

            # Handle special syntax like x__gt, x__lt, x__between
            if "__" in key:
                coord, op = key.rsplit("__", 1)
                coord_idx = {"x": 0, "y": 1, "z": 2}.get(coord)
                if coord_idx is None:
                    raise ValueError(f"Unknown coordinate: {coord}")

                coord_values = coords[:, coord_idx]

                if op == "gt":
                    mask &= coord_values > value
                elif op == "lt":
                    mask &= coord_values < value
                elif op == "gte":
                    mask &= coord_values >= value
                elif op == "lte":
                    mask &= coord_values <= value
                elif op == "between":
                    low, high = value
                    mask &= (coord_values >= low) & (coord_values <= high)
                else:
                    raise ValueError(f"Unknown operator: {op}")

            elif key in ("x", "y", "z"):
                # Exact match with tolerance
                coord_idx = {"x": 0, "y": 1, "z": 2}[key]
                coord_values = coords[:, coord_idx]

                if callable(value):
                    # Custom predicate function
                    mask &= np.array([value(v) for v in coord_values], dtype=bool)
                else:
                    # Exact match with tolerance
                    mask &= np.abs(coord_values - value) < tol

        # Return indices where mask is True
        return np.where(mask)[0].astype(np.int64)

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
    """Lazy face selection query.

    Returns face identifiers as Nx2 array of (element_index, local_face_index) pairs.
    """

    predicates: dict[str, Any] = field(default_factory=dict)
    _on_elements: "ElementQuery | None" = None

    def and_where(self, **kwargs: Any) -> FaceQuery:
        """Intersect with additional predicate."""
        new_predicates = {**self.predicates, **kwargs}
        return FaceQuery(predicates=new_predicates, _on_elements=self._on_elements)

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        """Evaluate query, return Nx2 array of (element_idx, local_face_idx).

        Args:
            mesh: The Mesh object to evaluate against. Must be the Python Mesh
                  wrapper (from mops.mesh), not the raw Rust _CoreMesh.

        Returns:
            Nx2 array of face identifiers [element_index, local_face_index].
        """
        # Import here to avoid circular import
        from mops.mesh import Mesh as PythonMesh

        # Handle raw _CoreMesh by checking for face methods
        if not hasattr(mesh, "get_all_faces"):
            raise ValueError(
                "FaceQuery requires a mops.Mesh object with face extraction methods. "
                "Pass the Python Mesh wrapper, not the raw Rust mesh."
            )

        # Start with appropriate initial set based on predicates
        if self._on_elements is not None:
            # Get faces only on selected elements
            elem_indices = set(self._on_elements.evaluate(mesh._core if hasattr(mesh, "_core") else mesh))
            all_faces = mesh.get_all_faces()
            face_mask = np.isin(all_faces[:, 0], list(elem_indices))
            candidate_faces = all_faces[face_mask]
        elif self.predicates.get("on_boundary", False):
            # Start with boundary faces only
            candidate_faces = mesh.get_boundary_faces()
        else:
            # Start with all faces
            candidate_faces = mesh.get_all_faces()

        if len(candidate_faces) == 0:
            return np.array([], dtype=np.int64).reshape(0, 2)

        # Build mask for filtering
        n_faces = len(candidate_faces)
        mask = np.ones(n_faces, dtype=bool)

        # Default tolerance for coordinate comparisons
        tol = self.predicates.get("tol", 1e-10)
        normal_tol = self.predicates.get("normal_tol", 0.1)  # Radians for angle tolerance

        # Get face centroids for spatial predicates (computed lazily)
        centroids = None
        normals = None

        for key, value in self.predicates.items():
            if key in ("on_boundary", "tol", "normal_tol", "indices"):
                continue

            # Handle index-based selection
            if key == "indices":
                # Direct face index selection (Nx2 array expected)
                return np.array(value, dtype=np.int64).reshape(-1, 2)

            # Handle normal direction predicate
            if key == "normal":
                if normals is None:
                    # Compute normals for candidate faces
                    normals = np.zeros((n_faces, 3), dtype=np.float64)
                    for i, (elem_idx, local_face_idx) in enumerate(candidate_faces):
                        normals[i] = mesh.get_face_normal(elem_idx, local_face_idx)

                # Normalize target normal
                target = np.array(value, dtype=np.float64)
                target_norm = np.linalg.norm(target)
                if target_norm > 1e-14:
                    target = target / target_norm

                # Compute angle between normals using dot product
                # cos(angle) = dot(a, b) / (|a||b|), and normals are already unit vectors
                dots = np.sum(normals * target, axis=1)
                # Clamp to [-1, 1] for numerical stability
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.arccos(dots)

                mask &= angles < normal_tol
                continue

            # Handle spatial predicates on face centroids
            if centroids is None:
                centroids = np.zeros((n_faces, 3), dtype=np.float64)
                for i, (elem_idx, local_face_idx) in enumerate(candidate_faces):
                    centroids[i] = mesh.get_face_centroid(elem_idx, local_face_idx)

            # Handle special syntax like x__gt, x__lt, x__between
            if "__" in key:
                coord, op = key.rsplit("__", 1)
                coord_idx = {"x": 0, "y": 1, "z": 2}.get(coord)
                if coord_idx is None:
                    raise ValueError(f"Unknown coordinate: {coord}")

                coord_values = centroids[:, coord_idx]

                if op == "gt":
                    mask &= coord_values > value
                elif op == "lt":
                    mask &= coord_values < value
                elif op == "gte":
                    mask &= coord_values >= value
                elif op == "lte":
                    mask &= coord_values <= value
                elif op == "between":
                    low, high = value
                    mask &= (coord_values >= low) & (coord_values <= high)
                else:
                    raise ValueError(f"Unknown operator: {op}")

            elif key in ("x", "y", "z"):
                # Exact match with tolerance on face centroid
                coord_idx = {"x": 0, "y": 1, "z": 2}[key]
                coord_values = centroids[:, coord_idx]

                if callable(value):
                    # Custom predicate function
                    mask &= np.array([value(v) for v in coord_values], dtype=bool)
                else:
                    # Exact match with tolerance
                    mask &= np.abs(coord_values - value) < tol

        # Return filtered faces
        return candidate_faces[mask]

    def __repr__(self) -> str:
        if self._on_elements is not None:
            return f"Faces.on_elements({self._on_elements}).and_where({self.predicates})"
        return f"Faces.where({self.predicates})"


class Faces:
    """Query builder for surface face selection.

    Faces are element boundary surfaces, essential for applying surface loads
    (pressure) and extracting boundary conditions.

    Example::

        # Boundary faces only
        Faces.on_boundary()

        # Faces with specific normal direction
        Faces.where(normal=(1, 0, 0))

        # Faces at specific coordinate
        Faces.where(x=100)

        # Faces on selected elements
        Faces.on_elements(Elements.where(material="steel"))

        # Combine predicates
        Faces.on_boundary().and_where(normal=(0, 0, 1))
    """

    @classmethod
    def on_boundary(cls) -> FaceQuery:
        """Select all boundary faces (not shared between elements).

        A face is on the boundary if it appears only once in the mesh,
        meaning it is not shared by two elements.

        Returns:
            FaceQuery selecting boundary faces.
        """
        return FaceQuery(predicates={"on_boundary": True})

    @classmethod
    def on_elements(cls, element_query: ElementQuery) -> FaceQuery:
        """Select faces on selected elements.

        Args:
            element_query: Query defining which elements to get faces from.

        Returns:
            FaceQuery selecting faces on the specified elements.
        """
        return FaceQuery(_on_elements=element_query)

    @classmethod
    def where(
        cls,
        normal: tuple[float, float, float] | None = None,
        normal_tol: float = 0.1,
        on_boundary: bool | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        x__gt: float | None = None,
        x__lt: float | None = None,
        x__gte: float | None = None,
        x__lte: float | None = None,
        x__between: tuple[float, float] | None = None,
        y__gt: float | None = None,
        y__lt: float | None = None,
        y__gte: float | None = None,
        y__lte: float | None = None,
        y__between: tuple[float, float] | None = None,
        z__gt: float | None = None,
        z__lt: float | None = None,
        z__gte: float | None = None,
        z__lte: float | None = None,
        z__between: tuple[float, float] | None = None,
        tol: float = 1e-10,
        **kwargs: Any,
    ) -> FaceQuery:
        """Select faces matching predicates.

        Predicates are applied to face centroids for spatial selection
        and face normals for normal direction matching.

        Args:
            normal: Face normal direction (unit vector, will match with tolerance).
            normal_tol: Angle tolerance in radians for normal matching (default 0.1).
            on_boundary: If True, select only boundary faces.
            x, y, z: Face centroid at specific coordinate (with tolerance).
            x__gt, y__gt, z__gt: Centroid coordinate greater than value.
            x__lt, y__lt, z__lt: Centroid coordinate less than value.
            x__gte, y__gte, z__gte: Centroid coordinate greater than or equal.
            x__lte, y__lte, z__lte: Centroid coordinate less than or equal.
            x__between, y__between, z__between: Centroid in range (inclusive).
            tol: Tolerance for exact coordinate matching (default 1e-10).
            **kwargs: Additional predicates.

        Returns:
            Lazy FaceQuery for evaluation.
        """
        predicates: dict[str, Any] = {}

        if normal is not None:
            predicates["normal"] = normal
        predicates["normal_tol"] = normal_tol
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
        if x__gte is not None:
            predicates["x__gte"] = x__gte
        if x__lte is not None:
            predicates["x__lte"] = x__lte
        if x__between is not None:
            predicates["x__between"] = x__between
        if y__gt is not None:
            predicates["y__gt"] = y__gt
        if y__lt is not None:
            predicates["y__lt"] = y__lt
        if y__gte is not None:
            predicates["y__gte"] = y__gte
        if y__lte is not None:
            predicates["y__lte"] = y__lte
        if y__between is not None:
            predicates["y__between"] = y__between
        if z__gt is not None:
            predicates["z__gt"] = z__gt
        if z__lt is not None:
            predicates["z__lt"] = z__lt
        if z__gte is not None:
            predicates["z__gte"] = z__gte
        if z__lte is not None:
            predicates["z__lte"] = z__lte
        if z__between is not None:
            predicates["z__between"] = z__between
        predicates["tol"] = tol

        predicates.update(kwargs)
        return FaceQuery(predicates=predicates)
