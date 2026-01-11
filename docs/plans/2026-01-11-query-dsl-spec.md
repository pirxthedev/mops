# Query DSL Specification

**Date:** 2026-01-11
**Status:** Draft
**Depends on:** [Python API Design](2026-01-10-python-api-design.md)
**Related issue:** mops-563, mops-2u1

## Overview

The Query DSL provides a declarative, Pythonic interface for selecting nodes, elements, and faces in finite element meshes. The design draws inspiration from ANSYS APDL selection commands (NSEL, ESEL, CM, CMSEL, NSLE, ESLN) while providing a modern, type-safe Python API.

### Design Principles

1. **Lazy Evaluation** - Queries are symbolic until evaluated against a mesh
2. **Composability** - Queries can be combined with set operations (union, intersection, difference)
3. **Immutability** - All query objects are immutable; methods return new instances
4. **Explicit over Implicit** - Selection criteria are clear and readable
5. **Pythonic Syntax** - Use Django/SQLAlchemy-style `__` operators for comparisons

## APDL Inspiration

The design is informed by ANSYS APDL selection commands, adapted for Python:

| APDL Command | MOPS Equivalent | Description |
|--------------|-----------------|-------------|
| `NSEL,S,LOC,X,0` | `Nodes.where(x=0)` | Select nodes at x=0 |
| `NSEL,S,LOC,X,0,10` | `Nodes.where(x__between=(0,10))` | Select nodes in range |
| `NSEL,R,LOC,Y,5` | `query.and_where(y=5)` | Reselect (intersection) |
| `NSEL,A,NODE,,1,7` | `query.union(Nodes.by_indices(...))` | Add to selection |
| `NSEL,U,LOC,Z,0` | `query.subtract(Nodes.where(z=0))` | Unselect |
| `NSEL,INVE` | `query.invert()` | Invert selection |
| `ESEL,S,TYPE,,1` | `Elements.where(type="tet4")` | Select by element type |
| `ESEL,S,MAT,,2` | `Elements.where(material="steel")` | Select by material |
| `ESLN,S,1` | `Elements.attached_to(node_query)` | Elements from nodes |
| `NSLE,S,ALL` | `Nodes.on_elements(elem_query)` | Nodes from elements |
| `CM,mycomp,NODE` | `model.define_component("mycomp", query)` | Create named group |
| `CMSEL,S,mycomp` | `Nodes.in_component("mycomp")` | Select component |

## Core Query Classes

### Query Base Class

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from mops import Mesh, Model

class Query(ABC):
    """Abstract base class for all queries."""

    @abstractmethod
    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        """Evaluate query against mesh, return indices as int64 array."""
        ...

    def union(self, other: "Query") -> "Query":
        """Union of two queries (set OR)."""
        return UnionQuery(self, other)

    def intersect(self, other: "Query") -> "Query":
        """Intersection of two queries (set AND)."""
        return IntersectQuery(self, other)

    def subtract(self, other: "Query") -> "Query":
        """Set difference (self - other)."""
        return SubtractQuery(self, other)

    def invert(self) -> "Query":
        """Invert selection (all entities not in this query)."""
        return InvertQuery(self)

    # Operator overloads for convenience
    def __or__(self, other: "Query") -> "Query":
        return self.union(other)

    def __and__(self, other: "Query") -> "Query":
        return self.intersect(other)

    def __sub__(self, other: "Query") -> "Query":
        return self.subtract(other)

    def __invert__(self) -> "Query":
        return self.invert()
```

### Set Operation Queries

```python
@dataclass(frozen=True)
class UnionQuery(Query):
    """Union of two queries."""
    left: Query
    right: Query

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        left_idx = self.left.evaluate(mesh)
        right_idx = self.right.evaluate(mesh)
        return np.unique(np.concatenate([left_idx, right_idx]))

@dataclass(frozen=True)
class IntersectQuery(Query):
    """Intersection of two queries."""
    left: Query
    right: Query

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        left_idx = set(self.left.evaluate(mesh))
        right_idx = set(self.right.evaluate(mesh))
        return np.array(sorted(left_idx & right_idx), dtype=np.int64)

@dataclass(frozen=True)
class SubtractQuery(Query):
    """Set difference of two queries."""
    left: Query
    right: Query

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        left_idx = set(self.left.evaluate(mesh))
        right_idx = set(self.right.evaluate(mesh))
        return np.array(sorted(left_idx - right_idx), dtype=np.int64)

@dataclass(frozen=True)
class InvertQuery(Query):
    """Inverted selection (complement)."""
    inner: Query

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        # Subclasses must override to know the universe set
        raise NotImplementedError("Must be specialized for entity type")
```

## Node Queries

### Nodes Class (Query Builder)

```python
class Nodes:
    """Query builder for node selection.

    Examples:
        # All nodes
        Nodes.all()

        # Nodes at specific coordinate
        Nodes.where(x=0)

        # Nodes with x > 10
        Nodes.where(x__gt=10)

        # Nodes in range (inclusive)
        Nodes.where(x__between=(0, 100))

        # Nodes by index
        Nodes.by_indices([0, 1, 2, 3])

        # Nodes near a point (within tolerance)
        Nodes.near_point((50, 25, 0), tol=0.1)

        # Nodes on elements
        Nodes.on_elements(Elements.where(material="steel"))

        # Nodes in named component
        Nodes.in_component("fixed_face")
    """

    @classmethod
    def all(cls) -> "NodeQuery":
        """Select all nodes."""
        return NodeQuery(_all=True)

    @classmethod
    def where(
        cls,
        # Exact coordinate match (with tolerance)
        x: float | Callable[[float], bool] | None = None,
        y: float | Callable[[float], bool] | None = None,
        z: float | Callable[[float], bool] | None = None,
        # Comparison operators
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
        # Tolerance for exact matches
        tol: float = 1e-10,
    ) -> "NodeQuery":
        """Select nodes matching spatial predicates."""
        ...

    @classmethod
    def by_indices(cls, indices: list[int] | np.ndarray) -> "NodeQuery":
        """Select nodes by explicit indices."""
        ...

    @classmethod
    def near_point(
        cls,
        point: tuple[float, float, float],
        tol: float,
    ) -> "NodeQuery":
        """Select nodes within tolerance of a point."""
        ...

    @classmethod
    def near_line(
        cls,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        tol: float,
    ) -> "NodeQuery":
        """Select nodes within tolerance of a line segment."""
        ...

    @classmethod
    def in_sphere(
        cls,
        center: tuple[float, float, float],
        radius: float,
    ) -> "NodeQuery":
        """Select nodes within a sphere."""
        ...

    @classmethod
    def in_box(
        cls,
        min_corner: tuple[float, float, float],
        max_corner: tuple[float, float, float],
    ) -> "NodeQuery":
        """Select nodes within an axis-aligned bounding box."""
        ...

    @classmethod
    def on_elements(cls, query: "ElementQuery") -> "NodeQuery":
        """Select nodes attached to selected elements (like APDL NSLE)."""
        ...

    @classmethod
    def in_component(cls, name: str) -> "NodeQuery":
        """Select nodes in a named component (like APDL CMSEL)."""
        ...
```

### NodeQuery Class

```python
@dataclass
class NodeQuery(Query):
    """Lazy node selection query.

    Stores predicates and evaluates them when needed against a mesh.
    """
    predicates: dict[str, Any] = field(default_factory=dict)
    _all: bool = False
    _on_elements: ElementQuery | None = None
    _component_name: str | None = None

    def and_where(self, **kwargs) -> "NodeQuery":
        """Intersect with additional predicate (like APDL NSEL,R)."""
        new_predicates = {**self.predicates, **kwargs}
        return NodeQuery(predicates=new_predicates)

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        """Evaluate query against mesh, return node indices."""
        ...
```

## Element Queries

### Elements Class (Query Builder)

```python
class Elements:
    """Query builder for element selection.

    Examples:
        # All elements
        Elements.all()

        # Elements by type
        Elements.where(type="tet10")

        # Elements with specific material
        Elements.where(material="steel")

        # Elements in named component
        Elements.in_component("bracket")

        # Elements attached to nodes
        Elements.attached_to(Nodes.where(x=0))

        # Elements by centroid location
        Elements.where(centroid_x__gt=50)
    """

    @classmethod
    def all(cls) -> "ElementQuery":
        """Select all elements."""
        return ElementQuery(_all=True)

    @classmethod
    def where(
        cls,
        # Element properties
        type: str | None = None,  # "tet4", "tet10", "hex8", etc.
        material: str | None = None,
        # Centroid-based spatial selection
        centroid_x: float | None = None,
        centroid_y: float | None = None,
        centroid_z: float | None = None,
        centroid_x__gt: float | None = None,
        centroid_x__lt: float | None = None,
        centroid_y__gt: float | None = None,
        centroid_y__lt: float | None = None,
        centroid_z__gt: float | None = None,
        centroid_z__lt: float | None = None,
    ) -> "ElementQuery":
        """Select elements matching predicates."""
        ...

    @classmethod
    def by_indices(cls, indices: list[int] | np.ndarray) -> "ElementQuery":
        """Select elements by explicit indices."""
        ...

    @classmethod
    def attached_to(cls, node_query: "NodeQuery") -> "ElementQuery":
        """Select elements attached to selected nodes (like APDL ESLN).

        Args:
            node_query: Query defining the nodes

        Returns:
            ElementQuery selecting elements where ALL nodes are in the selection
        """
        ...

    @classmethod
    def touching(cls, node_query: "NodeQuery") -> "ElementQuery":
        """Select elements touching selected nodes.

        Args:
            node_query: Query defining the nodes

        Returns:
            ElementQuery selecting elements where ANY node is in the selection
        """
        ...

    @classmethod
    def in_component(cls, name: str) -> "ElementQuery":
        """Select elements in a named component."""
        ...

    @classmethod
    def adjacent_to(cls, element_query: "ElementQuery") -> "ElementQuery":
        """Select elements adjacent (sharing face/edge) to selected elements."""
        ...
```

### ElementQuery Class

```python
@dataclass
class ElementQuery(Query):
    """Lazy element selection query."""
    predicates: dict[str, Any] = field(default_factory=dict)
    _all: bool = False
    _attached_to: NodeQuery | None = None
    _touching: NodeQuery | None = None
    _component_name: str | None = None

    def and_where(self, **kwargs) -> "ElementQuery":
        """Intersect with additional predicate."""
        ...

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        """Evaluate query against mesh, return element indices."""
        ...
```

## Face Queries

Faces are element boundary surfaces, essential for applying surface loads (pressure) and extracting boundary conditions.

### Faces Class (Query Builder)

```python
class Faces:
    """Query builder for surface face selection.

    Faces are defined by (element_index, local_face_index) pairs.

    Examples:
        # Boundary faces only
        Faces.on_boundary()

        # Faces with specific normal direction
        Faces.where(normal=(1, 0, 0), tol=0.1)

        # Faces at specific coordinate
        Faces.where(x=100)

        # Faces on selected elements
        Faces.on_elements(Elements.where(material="steel"))
    """

    @classmethod
    def on_boundary(cls) -> "FaceQuery":
        """Select all boundary faces (not shared between elements)."""
        return FaceQuery(predicates={"on_boundary": True})

    @classmethod
    def where(
        cls,
        # Face centroid spatial predicates
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        x__gt: float | None = None,
        x__lt: float | None = None,
        x__between: tuple[float, float] | None = None,
        y__gt: float | None = None,
        y__lt: float | None = None,
        y__between: tuple[float, float] | None = None,
        z__gt: float | None = None,
        z__lt: float | None = None,
        z__between: tuple[float, float] | None = None,
        # Normal direction
        normal: tuple[float, float, float] | None = None,
        normal_tol: float = 0.1,  # Angle tolerance in radians
        # Boundary flag
        on_boundary: bool | None = None,
        tol: float = 1e-10,
    ) -> "FaceQuery":
        """Select faces matching predicates."""
        ...

    @classmethod
    def on_elements(cls, query: "ElementQuery") -> "FaceQuery":
        """Select faces on selected elements."""
        ...

    @classmethod
    def in_component(cls, name: str) -> "FaceQuery":
        """Select faces in a named component."""
        ...
```

### FaceQuery Class

```python
@dataclass
class FaceQuery(Query):
    """Lazy face selection query.

    Returns face identifiers as (element_index, local_face_index) pairs.
    """
    predicates: dict[str, Any] = field(default_factory=dict)
    _on_elements: ElementQuery | None = None
    _component_name: str | None = None

    def and_where(self, **kwargs) -> "FaceQuery":
        """Intersect with additional predicate."""
        ...

    def evaluate(self, mesh: "Mesh") -> np.ndarray:
        """Evaluate query, return Nx2 array of (element_idx, face_idx)."""
        ...
```

## Named Components

Named components (groups) provide reusable selections, similar to APDL's CM command.

### Component Definition

```python
# Components are defined on the Model
model = (
    Model(mesh, materials={"steel": steel})
    .define_component("fixed_face", Nodes.where(x=0))
    .define_component("load_surface", Faces.where(x=100).and_where(on_boundary=True))
    .define_component("bracket_elements", Elements.where(centroid_z__gt=50))
)

# Components can be used in subsequent queries
model = model.constrain(Nodes.in_component("fixed_face"), dofs=["ux", "uy", "uz"])
model = model.load(Faces.in_component("load_surface"), Pressure(1e6))
```

### Component Storage

```python
@dataclass
class Component:
    """Named selection component."""
    name: str
    entity_type: Literal["node", "element", "face"]
    query: Query
    # Cached indices (evaluated lazily)
    _indices: np.ndarray | None = None

    def get_indices(self, mesh: "Mesh") -> np.ndarray:
        """Get (cached) indices for this component."""
        if self._indices is None:
            self._indices = self.query.evaluate(mesh)
        return self._indices
```

## Predicate Syntax

### Comparison Operators

Following Django/SQLAlchemy conventions:

| Suffix | Operation | Example |
|--------|-----------|---------|
| (none) | Exact equality (with tolerance) | `x=0` |
| `__gt` | Greater than | `x__gt=10` |
| `__lt` | Less than | `x__lt=50` |
| `__gte` | Greater than or equal | `x__gte=0` |
| `__lte` | Less than or equal | `x__lte=100` |
| `__between` | Inclusive range | `x__between=(0, 100)` |
| `__in` | In set of values | `type__in=["tet4", "tet10"]` |

### Tolerance Handling

Floating-point comparisons use configurable tolerance:

```python
# Default tolerance: 1e-10
Nodes.where(x=0)  # Matches nodes where |x| < 1e-10

# Custom tolerance
Nodes.where(x=0, tol=1e-6)  # More lenient matching

# For ranges, tolerance applies to boundaries
Nodes.where(x__between=(0, 100), tol=1e-6)
# Equivalent to: -tol <= x <= 100+tol
```

### Callable Predicates

For complex selection criteria, pass a callable:

```python
# Select nodes where x^2 + y^2 < r^2
Nodes.where(x=lambda x, y, z: x**2 + y**2 < 100**2)

# Select nodes in a custom region
def in_annulus(x, y, z):
    r = np.sqrt(x**2 + y**2)
    return (50 < r) & (r < 100)

Nodes.where(predicate=in_annulus)
```

## Complex Query Examples

### Example 1: Cantilever Beam Boundary Conditions

```python
# Fixed end at x=0
fixed_nodes = Nodes.where(x=0)
model = model.constrain(fixed_nodes, dofs=["ux", "uy", "uz"])

# Point load at tip
tip_node = Nodes.where(x=100, y=0, z=0)
model = model.load(tip_node, Force(fy=-1000))
```

### Example 2: NAFEMS LE1 Elliptic Membrane

```python
# Inner ellipse (semi-axes a=1000, b=520): constrained in x
inner_x = Nodes.where(y=0, x__gt=0, x__lte=1000)
model = model.constrain(inner_x, dofs=["ux"])

# Inner ellipse: constrained in y
inner_y = Nodes.where(x=0, y__gt=0, y__lte=520)
model = model.constrain(inner_y, dofs=["uy"])

# Symmetry at y=0 (x-axis)
sym_y = Nodes.where(y=0)
model = model.constrain(sym_y, dofs=["uy"])

# Symmetry at x=0 (y-axis)
sym_x = Nodes.where(x=0)
model = model.constrain(sym_x, dofs=["ux"])

# Outer ellipse: uniform tension
outer_faces = Faces.on_boundary().and_where(
    predicate=lambda cx, cy, cz: is_on_outer_ellipse(cx, cy)
)
```

### Example 3: Set Operations

```python
# Select all boundary nodes except those on the fixed face
free_boundary = Nodes.on_boundary() - Nodes.where(x=0)

# Select nodes in either region
combined = Nodes.where(x=0) | Nodes.where(x=100)

# Select nodes in both regions (intersection)
corner = Nodes.where(x=0).and_where(y=0)

# Using operator overloads
region1 = Nodes.where(x__between=(0, 50))
region2 = Nodes.where(y__between=(0, 50))
overlap = region1 & region2  # Intersection
combined = region1 | region2  # Union
```

### Example 4: Topological Selection

```python
# Select elements attached to fixed nodes
fixed_elements = Elements.attached_to(Nodes.where(x=0))

# Select nodes on those elements (expand selection)
expanded_nodes = Nodes.on_elements(fixed_elements)

# Select boundary faces on specific elements
bracket_faces = Faces.on_elements(
    Elements.where(material="bracket")
).and_where(on_boundary=True)
```

## Implementation Notes

### Evaluation Strategy

1. **Lazy evaluation**: Queries store predicates but don't compute indices until `evaluate()` is called
2. **Caching**: Mesh can cache expensive computations (element centroids, face normals, adjacency)
3. **Short-circuit**: `all()` and `none()` queries avoid iteration
4. **Vectorized operations**: Use NumPy broadcasting for spatial predicates

### Performance Considerations

```python
class Mesh:
    # Cached computed properties
    @cached_property
    def element_centroids(self) -> np.ndarray:
        """Nx3 array of element centroid coordinates."""
        ...

    @cached_property
    def boundary_faces(self) -> np.ndarray:
        """Kx2 array of boundary (element_idx, face_idx) pairs."""
        ...

    @cached_property
    def face_normals(self) -> dict[tuple[int, int], np.ndarray]:
        """Map from (element_idx, face_idx) to unit normal vector."""
        ...

    @cached_property
    def node_to_elements(self) -> dict[int, list[int]]:
        """Map from node index to element indices."""
        ...
```

### Error Handling

```python
class QueryError(MopsError):
    """Error during query evaluation."""
    pass

class EmptyQueryError(QueryError):
    """Query matched no entities."""
    pass

class AmbiguousQueryError(QueryError):
    """Query matched unexpected number of entities."""
    pass
```

Queries that match zero entities should raise `EmptyQueryError` with helpful context:

```python
def evaluate(self, mesh: Mesh) -> np.ndarray:
    indices = self._compute_indices(mesh)
    if len(indices) == 0:
        raise EmptyQueryError(
            f"Query {self} matched no nodes in mesh with {mesh.n_nodes} nodes. "
            f"Coordinate bounds: {mesh.bounds}"
        )
    return indices
```

## Testing Strategy

### Unit Tests

1. **Predicate evaluation**: Test each predicate type with known meshes
2. **Set operations**: Test union, intersection, difference, inversion
3. **Edge cases**: Empty meshes, single-element meshes, degenerate cases
4. **Tolerance handling**: Verify floating-point comparisons

### Integration Tests

1. **Full pipeline**: Query → Model → Solve flow
2. **Component definitions**: Named groups across model operations
3. **Complex queries**: Multi-level compositions

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=3, max_size=3))
def test_point_in_sphere_query(center):
    # Generated test cases for spatial queries
    ...
```

## Open Questions

1. **Face indexing**: Should faces be (element, local_face) or global face IDs?
2. **Lazy vs eager component evaluation**: When should component indices be cached?
3. **Query serialization**: Support for saving/loading queries?
4. **Parallel evaluation**: Thread-safe query evaluation for large meshes?

## References

- [ANSYS APDL NSEL Command](https://www.mm.bme.hu/~gyebro/files/ans_help_v182/ans_cmd/Hlp_C_NSEL.html)
- [ANSYS APDL ESEL Command](https://www.mm.bme.hu/~gyebro/files/ans_help_v182/ans_cmd/Hlp_C_ESEL.html)
- [ANSYS APDL CMSEL Command](https://www.mm.bme.hu/~gyebro/files/ans_help_v182/ans_cmd/Hlp_C_CMSEL.html)
- [Django QuerySet API](https://docs.djangoproject.com/en/4.2/ref/models/querysets/)
- [SQLAlchemy Core Expression Language](https://docs.sqlalchemy.org/en/20/core/tutorial.html)
