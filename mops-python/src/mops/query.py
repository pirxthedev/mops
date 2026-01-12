"""Query DSL for node, element, and face selection.

The query system provides a declarative way to select parts of the mesh
for applying materials, constraints, and loads.

Example::

    # Select all nodes at x=0
    Nodes.where(x=0)

    # Select nodes in a range
    Nodes.where(x__gt=0, x__lt=10)

    # Combine queries with methods
    Nodes.where(x=0).union(Nodes.where(x=100))
    Nodes.where(x__gt=0).intersect(Nodes.where(y__gt=0))
    Nodes.where(x__between=(0, 100)).subtract(Nodes.where(y=0))

    # Or use operators for cleaner syntax
    Nodes.where(x=0) | Nodes.where(x=100)     # union
    Nodes.where(x__gt=0) & Nodes.where(y__gt=0)  # intersection
    Nodes.where(x__between=(0, 100)) - Nodes.where(y=0)  # subtract
    ~Nodes.where(x=0)  # invert (all nodes NOT at x=0)

    # Use named components
    model = model.define_component("fixed_face", Nodes.where(x=0))
    Nodes.in_component("fixed_face")
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
    def evaluate(self, mesh: "Mesh", components: dict[str, "Query"] | None = None) -> np.ndarray:
        """Evaluate query against mesh, return indices.

        Args:
            mesh: The mesh to evaluate against
            components: Optional dict of named components for resolving component queries

        Returns:
            Array of indices (int64)
        """
        ...

    def union(self, other: Query) -> Query:
        """Union of two queries (A ∪ B).

        Args:
            other: Query to union with

        Returns:
            Query that selects items in either query
        """
        return UnionQuery(self, other)

    def intersect(self, other: Query) -> Query:
        """Intersection of two queries (A ∩ B).

        Args:
            other: Query to intersect with

        Returns:
            Query that selects items in both queries
        """
        return IntersectQuery(self, other)

    def subtract(self, other: Query) -> Query:
        """Set difference of two queries (A - B).

        Args:
            other: Query to subtract

        Returns:
            Query that selects items in self but not in other
        """
        return SubtractQuery(self, other)

    def invert(self) -> Query:
        """Invert query to select everything NOT in current selection (~A).

        Returns:
            Query that selects items not in the current query
        """
        return InvertQuery(self)

    # Operator overloads for more natural syntax
    def __or__(self, other: Query) -> Query:
        """Union operator: q1 | q2"""
        return self.union(other)

    def __and__(self, other: Query) -> Query:
        """Intersection operator: q1 & q2"""
        return self.intersect(other)

    def __sub__(self, other: Query) -> Query:
        """Subtract operator: q1 - q2"""
        return self.subtract(other)

    def __invert__(self) -> Query:
        """Invert operator: ~q"""
        return self.invert()

    @property
    def component_name(self) -> str | None:
        """Return component name if this is a component query, else None."""
        return None


@dataclass
class UnionQuery(Query):
    """Union of two queries."""

    left: Query
    right: Query

    def evaluate(self, mesh: "Mesh", components: dict[str, "Query"] | None = None) -> np.ndarray:
        left_indices = self.left.evaluate(mesh, components)
        right_indices = self.right.evaluate(mesh, components)
        return np.unique(np.concatenate([left_indices, right_indices]))


@dataclass
class SubtractQuery(Query):
    """Set difference of two queries (A - B)."""

    left: Query
    right: Query

    def evaluate(self, mesh: "Mesh", components: dict[str, "Query"] | None = None) -> np.ndarray:
        left_indices = set(self.left.evaluate(mesh, components))
        right_indices = set(self.right.evaluate(mesh, components))
        return np.array(sorted(left_indices - right_indices), dtype=np.int64)


@dataclass
class IntersectQuery(Query):
    """Intersection of two queries (A ∩ B)."""

    left: Query
    right: Query

    def evaluate(self, mesh: "Mesh", components: dict[str, "Query"] | None = None) -> np.ndarray:
        left_indices = set(self.left.evaluate(mesh, components))
        right_indices = set(self.right.evaluate(mesh, components))
        return np.array(sorted(left_indices & right_indices), dtype=np.int64)


@dataclass
class InvertQuery(Query):
    """Inversion of a query (~A).

    Selects all items NOT in the inner query. The universe (all possible items)
    is determined by the inner query's type:
    - NodeQuery: all nodes in mesh
    - ElementQuery: all elements in mesh
    - FaceQuery: all faces in mesh
    """

    inner: Query

    def evaluate(self, mesh: "Mesh", components: dict[str, "Query"] | None = None) -> np.ndarray:
        inner_indices = set(self.inner.evaluate(mesh, components))

        # Determine the universe based on inner query type
        if isinstance(self.inner, (NodeQuery, UnionQuery, SubtractQuery, IntersectQuery, InvertQuery)):
            # For node queries (or composed queries), use all node indices
            # Need to detect the base query type
            universe = self._get_universe(mesh, components)
        else:
            universe = self._get_universe(mesh, components)

        return np.array(sorted(universe - inner_indices), dtype=np.int64)

    def _get_universe(self, mesh: "Mesh", components: dict[str, "Query"] | None = None) -> set[int]:
        """Determine the universe of all possible indices based on inner query type."""
        # Walk down to find the base query type
        base_query = self._find_base_query(self.inner)

        if isinstance(base_query, NodeQuery):
            # Import here to avoid circular import issues
            try:
                n_nodes = mesh.n_nodes
            except AttributeError:
                # Handle Python Mesh wrapper
                if hasattr(mesh, "_core"):
                    n_nodes = mesh._core.n_nodes
                else:
                    n_nodes = mesh.n_nodes
            return set(range(n_nodes))
        elif isinstance(base_query, ElementQuery):
            try:
                n_elements = mesh.n_elements
            except AttributeError:
                if hasattr(mesh, "_core"):
                    n_elements = mesh._core.n_elements
                else:
                    n_elements = mesh.n_elements
            return set(range(n_elements))
        elif isinstance(base_query, FaceQuery):
            # For faces, get all faces from mesh
            from mops.mesh import Mesh as PythonMesh
            if hasattr(mesh, "get_all_faces"):
                all_faces = mesh.get_all_faces()
                # Convert face array to set of tuples for comparison
                # Note: FaceQuery returns Nx2 array, but indices are linear for set ops
                return set(range(len(all_faces)))
            else:
                raise ValueError("Cannot invert FaceQuery: mesh doesn't support get_all_faces()")
        else:
            # Fallback: assume node query
            try:
                n_nodes = mesh.n_nodes
            except AttributeError:
                if hasattr(mesh, "_core"):
                    n_nodes = mesh._core.n_nodes
                else:
                    n_nodes = mesh.n_nodes
            return set(range(n_nodes))

    def _find_base_query(self, query: Query) -> Query:
        """Walk down composite queries to find the base query type."""
        if isinstance(query, (UnionQuery, SubtractQuery, IntersectQuery)):
            # Check left branch first
            return self._find_base_query(query.left)
        elif isinstance(query, InvertQuery):
            return self._find_base_query(query.inner)
        else:
            return query


@dataclass
class NodeQuery(Query):
    """Lazy node selection query.

    Stores predicates and evaluates them when needed.
    """

    predicates: dict[str, Any] = field(default_factory=dict)
    _all: bool = False
    _component_name: str | None = None
    _near_point: tuple[tuple[float, float, float], float] | None = None
    _in_sphere: tuple[tuple[float, float, float], float] | None = None
    _in_box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    _near_line: tuple[tuple[float, float, float], tuple[float, float, float], float] | None = None
    _on_elements: "ElementQuery | None" = None
    _physical_group: str | None = None

    def and_where(self, **kwargs: Any) -> NodeQuery:
        """Intersect with additional predicate."""
        new_predicates = {**self.predicates, **kwargs}
        return NodeQuery(predicates=new_predicates)

    @property
    def component_name(self) -> str | None:
        """Return component name if this is a component query, else None."""
        return self._component_name

    def evaluate(self, mesh: "Mesh", components: dict[str, "Query"] | None = None) -> np.ndarray:
        """Evaluate query, return node indices."""
        # Handle component query - resolve the referenced component
        if self._component_name is not None:
            if components is None:
                raise ValueError(
                    f"Component query '{self._component_name}' requires components dict. "
                    "Use model methods which pass components automatically."
                )
            if self._component_name not in components:
                raise ValueError(
                    f"Unknown component: '{self._component_name}'. "
                    f"Available: {list(components.keys())}"
                )
            component_query = components[self._component_name]
            # Verify the component query returns node indices (is a node query)
            if not isinstance(component_query, NodeQuery):
                raise TypeError(
                    f"Component '{self._component_name}' is not a node query. "
                    "Use the appropriate query type (Nodes, Elements, or Faces)."
                )
            return component_query.evaluate(mesh, components)

        # Handle physical group query - look up nodes from mesh's physical groups
        if self._physical_group is not None:
            # Import here to avoid circular import
            from mops.mesh import Mesh as PythonMesh
            if not isinstance(mesh, PythonMesh):
                raise TypeError(
                    f"Physical group queries require a Python Mesh object with physical groups. "
                    f"Got: {type(mesh)}"
                )
            group = mesh.get_physical_group(self._physical_group)
            return np.array(group.node_indices, dtype=np.int64)

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

        # Handle geometric queries (near_point, in_sphere, in_box, near_line)
        if self._near_point is not None:
            point, tol = self._near_point
            point_arr = np.array(point, dtype=np.float64)
            distances = np.linalg.norm(coords - point_arr, axis=1)
            return np.where(distances <= tol)[0].astype(np.int64)

        if self._in_sphere is not None:
            center, radius = self._in_sphere
            center_arr = np.array(center, dtype=np.float64)
            distances = np.linalg.norm(coords - center_arr, axis=1)
            return np.where(distances <= radius)[0].astype(np.int64)

        if self._in_box is not None:
            min_corner, max_corner = self._in_box
            min_arr = np.array(min_corner, dtype=np.float64)
            max_arr = np.array(max_corner, dtype=np.float64)
            in_box = np.all((coords >= min_arr) & (coords <= max_arr), axis=1)
            return np.where(in_box)[0].astype(np.int64)

        if self._near_line is not None:
            start, end, tol = self._near_line
            start_arr = np.array(start, dtype=np.float64)
            end_arr = np.array(end, dtype=np.float64)
            # Compute distance from each node to the line segment
            line_vec = end_arr - start_arr
            line_len_sq = np.dot(line_vec, line_vec)

            if line_len_sq < 1e-14:
                # Degenerate case: start == end, treat as point
                distances = np.linalg.norm(coords - start_arr, axis=1)
            else:
                # Project each point onto the line and clamp to segment
                # t = ((P - A) · (B - A)) / |B - A|^2, clamped to [0, 1]
                t = np.dot(coords - start_arr, line_vec) / line_len_sq
                t = np.clip(t, 0.0, 1.0)
                # Closest point on segment: A + t * (B - A)
                closest = start_arr + np.outer(t, line_vec)
                distances = np.linalg.norm(coords - closest, axis=1)

            return np.where(distances <= tol)[0].astype(np.int64)

        # Handle topological query: nodes on selected elements (like APDL NSLE)
        if self._on_elements is not None:
            # Get element connectivity from mesh
            if hasattr(mesh, "elements"):
                elements = mesh.elements
            elif hasattr(mesh, "_core"):
                elements = mesh._core.elements
            else:
                elements = mesh.elements

            # Get selected element indices
            elem_indices = self._on_elements.evaluate(mesh, components)
            if len(elem_indices) == 0:
                return np.array([], dtype=np.int64)

            # Collect all unique node indices from selected elements
            node_set: set[int] = set()
            for elem_idx in elem_indices:
                for node_idx in elements[elem_idx]:
                    node_set.add(int(node_idx))

            return np.array(sorted(node_set), dtype=np.int64)

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
        if self._component_name is not None:
            return f"Nodes.in_component('{self._component_name}')"
        if self._physical_group is not None:
            return f"Nodes.from_physical_group('{self._physical_group}')"
        if self._all:
            return "Nodes.all()"
        if self._near_point is not None:
            point, tol = self._near_point
            return f"Nodes.near_point({point}, tol={tol})"
        if self._in_sphere is not None:
            center, radius = self._in_sphere
            return f"Nodes.in_sphere({center}, radius={radius})"
        if self._in_box is not None:
            min_corner, max_corner = self._in_box
            return f"Nodes.in_box({min_corner}, {max_corner})"
        if self._near_line is not None:
            start, end, tol = self._near_line
            return f"Nodes.near_line({start}, {end}, tol={tol})"
        if self._on_elements is not None:
            return f"Nodes.on_elements({self._on_elements})"
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

    @classmethod
    def in_component(cls, name: str) -> NodeQuery:
        """Select nodes in a named component.

        Components are defined using Model.define_component() and allow
        reusable selections to be referenced by name, similar to APDL's
        CM (component manager) and CMSEL commands.

        Args:
            name: Name of the component to select

        Returns:
            NodeQuery that resolves to the component's nodes when evaluated

        Example::

            # Define a component
            model = model.define_component("fixed_nodes", Nodes.where(x=0))

            # Use the component in constraints
            model = model.constrain(Nodes.in_component("fixed_nodes"),
                                    dofs=["ux", "uy", "uz"])
        """
        return NodeQuery(_component_name=name)

    @classmethod
    def near_point(
        cls,
        point: tuple[float, float, float],
        tol: float,
    ) -> NodeQuery:
        """Select nodes within tolerance of a point.

        Args:
            point: (x, y, z) coordinates of the point
            tol: Maximum distance from the point to select nodes

        Returns:
            NodeQuery selecting nodes within distance tol of the point

        Example::

            # Select nodes near a specific point
            Nodes.near_point((50, 25, 0), tol=0.1)
        """
        return NodeQuery(_near_point=(point, tol))

    @classmethod
    def in_sphere(
        cls,
        center: tuple[float, float, float],
        radius: float,
    ) -> NodeQuery:
        """Select nodes within a sphere.

        Args:
            center: (x, y, z) coordinates of the sphere center
            radius: Radius of the sphere

        Returns:
            NodeQuery selecting nodes inside the sphere (distance <= radius)

        Example::

            # Select nodes in a spherical region
            Nodes.in_sphere((50, 50, 50), radius=10)
        """
        return NodeQuery(_in_sphere=(center, radius))

    @classmethod
    def in_box(
        cls,
        min_corner: tuple[float, float, float],
        max_corner: tuple[float, float, float],
    ) -> NodeQuery:
        """Select nodes within an axis-aligned bounding box.

        Args:
            min_corner: (x_min, y_min, z_min) - corner with minimum coordinates
            max_corner: (x_max, y_max, z_max) - corner with maximum coordinates

        Returns:
            NodeQuery selecting nodes inside the box (inclusive on all sides)

        Example::

            # Select nodes in a rectangular region
            Nodes.in_box((0, 0, 0), (100, 50, 25))
        """
        return NodeQuery(_in_box=(min_corner, max_corner))

    @classmethod
    def near_line(
        cls,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        tol: float,
    ) -> NodeQuery:
        """Select nodes within tolerance of a line segment.

        Args:
            start: (x, y, z) coordinates of the line segment start point
            end: (x, y, z) coordinates of the line segment end point
            tol: Maximum distance from the line segment to select nodes

        Returns:
            NodeQuery selecting nodes within distance tol of the line segment

        Example::

            # Select nodes near an edge
            Nodes.near_line((0, 0, 0), (100, 0, 0), tol=0.5)
        """
        return NodeQuery(_near_line=(start, end, tol))

    @classmethod
    def from_physical_group(cls, name: str) -> NodeQuery:
        """Select nodes from a Gmsh physical group.

        Physical groups are named collections of entities defined in Gmsh.
        When a mesh is imported from Gmsh via Mesh.from_gmsh(), physical
        groups are extracted and stored in the mesh.

        This method provides a convenient way to apply boundary conditions
        to named surfaces or regions without relying on geometric queries.

        Args:
            name: Name of the physical group (as defined in Gmsh)

        Returns:
            NodeQuery selecting all nodes in the physical group

        Raises:
            KeyError: If the physical group name is not found in the mesh

        Example::

            # In Gmsh, define physical groups:
            # gmsh.model.addPhysicalGroup(2, [surface_tags], name="fixed")
            # gmsh.model.addPhysicalGroup(2, [surface_tags], name="loaded")

            mesh = Mesh.from_gmsh(gmsh.model)

            # Use physical groups for boundary conditions
            model = (
                Model(mesh, materials={"steel": steel})
                .assign(Elements.all(), material="steel")
                .constrain(Nodes.from_physical_group("fixed"), dofs=["ux", "uy", "uz"])
                .load(Nodes.from_physical_group("loaded"), Force(fy=-1000))
            )
        """
        return NodeQuery(_physical_group=name)

    @classmethod
    def on_elements(cls, element_query: "ElementQuery") -> NodeQuery:
        """Select nodes on selected elements (like APDL NSLE).

        Returns all nodes that belong to any of the selected elements.
        This is useful for expanding an element selection to include all
        associated nodes.

        Args:
            element_query: Query defining which elements to get nodes from

        Returns:
            NodeQuery selecting all nodes on the specified elements

        Example::

            # Get all nodes on elements at the fixed boundary
            fixed_elem_nodes = Nodes.on_elements(Elements.attached_to(Nodes.where(x=0)))

            # Expand element selection to include all nodes
            bracket_nodes = Nodes.on_elements(Elements.where(material="bracket"))
        """
        return NodeQuery(_on_elements=element_query)


@dataclass
class ElementQuery(Query):
    """Lazy element selection query."""

    predicates: dict[str, Any] = field(default_factory=dict)
    _all: bool = False
    _component_name: str | None = None
    _attached_to: "NodeQuery | None" = None
    _touching: "NodeQuery | None" = None
    _physical_group: str | None = None

    def and_where(self, **kwargs: Any) -> ElementQuery:
        """Intersect with additional predicate."""
        new_predicates = {**self.predicates, **kwargs}
        return ElementQuery(predicates=new_predicates)

    @property
    def component_name(self) -> str | None:
        """Return component name if this is a component query, else None."""
        return self._component_name

    def evaluate(self, mesh: "Mesh", components: dict[str, "Query"] | None = None) -> np.ndarray:
        """Evaluate query, return element indices."""
        # Handle component query - resolve the referenced component
        if self._component_name is not None:
            if components is None:
                raise ValueError(
                    f"Component query '{self._component_name}' requires components dict. "
                    "Use model methods which pass components automatically."
                )
            if self._component_name not in components:
                raise ValueError(
                    f"Unknown component: '{self._component_name}'. "
                    f"Available: {list(components.keys())}"
                )
            component_query = components[self._component_name]
            # Verify the component query returns element indices (is an element query)
            if not isinstance(component_query, ElementQuery):
                raise TypeError(
                    f"Component '{self._component_name}' is not an element query. "
                    "Use the appropriate query type (Nodes, Elements, or Faces)."
                )
            return component_query.evaluate(mesh, components)

        # Handle physical group query - look up elements from mesh's physical groups
        if self._physical_group is not None:
            # Import here to avoid circular import
            from mops.mesh import Mesh as PythonMesh
            if not isinstance(mesh, PythonMesh):
                raise TypeError(
                    f"Physical group queries require a Python Mesh object with physical groups. "
                    f"Got: {type(mesh)}"
                )
            group = mesh.get_physical_group(self._physical_group)
            return np.array(group.element_indices, dtype=np.int64)

        # Get element connectivity from mesh
        # Handle both Python Mesh wrapper and raw Rust mesh
        if hasattr(mesh, "elements"):
            elements = mesh.elements
            n_elements = mesh.n_elements
        elif hasattr(mesh, "_core"):
            elements = mesh._core.elements
            n_elements = mesh._core.n_elements
        else:
            elements = mesh.elements
            n_elements = mesh.n_elements

        # Handle topological queries
        if self._attached_to is not None:
            # Elements where ALL nodes are in the node selection (like APDL ESLN)
            # Get selected node indices
            node_indices = set(self._attached_to.evaluate(mesh, components))
            if len(node_indices) == 0:
                return np.array([], dtype=np.int64)

            selected = []
            for elem_idx in range(n_elements):
                elem_nodes = set(elements[elem_idx])
                # Element is selected if ALL its nodes are in the selection
                if elem_nodes <= node_indices:
                    selected.append(elem_idx)
            return np.array(selected, dtype=np.int64)

        if self._touching is not None:
            # Elements where ANY node is in the node selection
            # Get selected node indices
            node_indices = set(self._touching.evaluate(mesh, components))
            if len(node_indices) == 0:
                return np.array([], dtype=np.int64)

            selected = []
            for elem_idx in range(n_elements):
                elem_nodes = set(elements[elem_idx])
                # Element is selected if ANY of its nodes are in the selection
                if elem_nodes & node_indices:
                    selected.append(elem_idx)
            return np.array(selected, dtype=np.int64)

        if self._all:
            return np.arange(n_elements, dtype=np.int64)

        # Apply predicates (simplified implementation)
        if "indices" in self.predicates:
            return np.array(self.predicates["indices"], dtype=np.int64)

        return np.arange(n_elements, dtype=np.int64)

    def __repr__(self) -> str:
        if self._all:
            return "Elements.all()"
        if self._component_name is not None:
            return f"Elements.in_component('{self._component_name}')"
        if self._physical_group is not None:
            return f"Elements.from_physical_group('{self._physical_group}')"
        if self._attached_to is not None:
            return f"Elements.attached_to({self._attached_to})"
        if self._touching is not None:
            return f"Elements.touching({self._touching})"
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

    @classmethod
    def in_component(cls, name: str) -> ElementQuery:
        """Select elements in a named component.

        Components are defined using Model.define_component() and allow
        reusable selections to be referenced by name, similar to APDL's
        CM (component manager) and CMSEL commands.

        Args:
            name: Name of the component to select

        Returns:
            ElementQuery that resolves to the component's elements when evaluated

        Example::

            # Define a component
            model = model.define_component("bracket", Elements.where(type="hex8"))

            # Use the component
            model = model.assign(Elements.in_component("bracket"), material="steel")
        """
        return ElementQuery(_component_name=name)

    @classmethod
    def attached_to(cls, node_query: NodeQuery) -> ElementQuery:
        """Select elements where ALL nodes are in the node selection.

        This is equivalent to APDL's ESLN command. An element is selected
        only if every node of that element is in the node query result.

        Args:
            node_query: Query defining which nodes elements must be attached to

        Returns:
            ElementQuery selecting elements fully contained in the node selection

        Example::

            # Select elements fully attached to fixed boundary
            fixed_elements = Elements.attached_to(Nodes.where(x=0))

            # Use with pressure loads - only elements fully on surface
            Elements.attached_to(Nodes.where(z=100))
        """
        return ElementQuery(_attached_to=node_query)

    @classmethod
    def touching(cls, node_query: NodeQuery) -> ElementQuery:
        """Select elements where ANY node is in the node selection.

        An element is selected if at least one of its nodes is in the
        node query result. This is useful for finding elements adjacent
        to a boundary or region.

        Args:
            node_query: Query defining which nodes elements may touch

        Returns:
            ElementQuery selecting elements with at least one node in the selection

        Example::

            # Select elements touching the boundary (any node at x=0)
            boundary_elements = Elements.touching(Nodes.where(x=0))

            # Elements in the vicinity of a point
            nearby = Elements.touching(Nodes.near_point((50, 50, 50), tol=5))
        """
        return ElementQuery(_touching=node_query)

    @classmethod
    def from_physical_group(cls, name: str) -> ElementQuery:
        """Select elements from a Gmsh physical group.

        Physical groups are named collections of entities defined in Gmsh.
        When a mesh is imported from Gmsh via Mesh.from_gmsh(), physical
        groups are extracted and stored in the mesh.

        This method provides a convenient way to assign materials to named
        volumes without relying on geometric queries.

        Args:
            name: Name of the physical group (as defined in Gmsh)

        Returns:
            ElementQuery selecting all elements in the physical group

        Raises:
            KeyError: If the physical group name is not found in the mesh

        Example::

            # In Gmsh, define physical groups for volumes:
            # gmsh.model.addPhysicalGroup(3, [volume_tags], name="bracket")
            # gmsh.model.addPhysicalGroup(3, [volume_tags], name="bolt")

            mesh = Mesh.from_gmsh(gmsh.model)

            # Use physical groups for material assignment
            model = (
                Model(mesh, materials={"steel": steel, "titanium": titanium})
                .assign(Elements.from_physical_group("bracket"), material="steel")
                .assign(Elements.from_physical_group("bolt"), material="titanium")
            )
        """
        return ElementQuery(_physical_group=name)


@dataclass
class FaceQuery(Query):
    """Lazy face selection query.

    Returns face identifiers as Nx2 array of (element_index, local_face_index) pairs.
    """

    predicates: dict[str, Any] = field(default_factory=dict)
    _on_elements: "ElementQuery | None" = None
    _component_name: str | None = None

    def and_where(self, **kwargs: Any) -> FaceQuery:
        """Intersect with additional predicate."""
        new_predicates = {**self.predicates, **kwargs}
        return FaceQuery(predicates=new_predicates, _on_elements=self._on_elements)

    @property
    def component_name(self) -> str | None:
        """Return component name if this is a component query, else None."""
        return self._component_name

    def evaluate(self, mesh: "Mesh", components: dict[str, "Query"] | None = None) -> np.ndarray:
        """Evaluate query, return Nx2 array of (element_idx, local_face_idx).

        Args:
            mesh: The Mesh object to evaluate against. Must be the Python Mesh
                  wrapper (from mops.mesh), not the raw Rust _CoreMesh.
            components: Optional dict of named components for resolving component queries

        Returns:
            Nx2 array of face identifiers [element_index, local_face_index].
        """
        # Handle component query - resolve the referenced component
        if self._component_name is not None:
            if components is None:
                raise ValueError(
                    f"Component query '{self._component_name}' requires components dict. "
                    "Use model methods which pass components automatically."
                )
            if self._component_name not in components:
                raise ValueError(
                    f"Unknown component: '{self._component_name}'. "
                    f"Available: {list(components.keys())}"
                )
            component_query = components[self._component_name]
            # Verify the component query returns face indices (is a face query)
            if not isinstance(component_query, FaceQuery):
                raise TypeError(
                    f"Component '{self._component_name}' is not a face query. "
                    "Use the appropriate query type (Nodes, Elements, or Faces)."
                )
            return component_query.evaluate(mesh, components)

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
        if self._component_name is not None:
            return f"Faces.in_component('{self._component_name}')"
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

    @classmethod
    def in_component(cls, name: str) -> FaceQuery:
        """Select faces in a named component.

        Components are defined using Model.define_component() and allow
        reusable selections to be referenced by name, similar to APDL's
        CM (component manager) and CMSEL commands.

        Args:
            name: Name of the component to select

        Returns:
            FaceQuery that resolves to the component's faces when evaluated

        Example::

            # Define a component
            model = model.define_component("load_surface",
                                           Faces.on_boundary().and_where(x=100))

            # Use the component
            model = model.load(Faces.in_component("load_surface"), Pressure(1e6))
        """
        return FaceQuery(_component_name=name)
