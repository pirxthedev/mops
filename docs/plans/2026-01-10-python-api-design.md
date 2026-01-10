# Python API Design

**Date:** 2026-01-10
**Status:** Draft
**Depends on:** [MOPS Initial Design](2026-01-10-mops-initial-design.md)

## Overview

The Python API provides a declarative, immutable interface for building FEA models. The design emphasizes:

1. **Immutability** - Models are frozen after construction
2. **Copy-on-write** - Method chaining returns new instances
3. **Pure functions** - `solve()` is a transformation, not mutation
4. **Query DSL** - Declarative selections for nodes/elements/faces

## Package Structure

```
mops-python/
├── pyproject.toml          # maturin build config
├── src/
│   └── mops/
│       ├── __init__.py     # Public API exports
│       ├── model.py        # Model class
│       ├── mesh.py         # Mesh + Gmsh integration
│       ├── material.py     # Material definitions
│       ├── loads.py        # Load types
│       ├── query.py        # Query DSL (Nodes, Elements, Faces)
│       ├── results.py      # Results access
│       └── _core.pyi       # Type stubs for Rust bindings
└── rust/
    └── src/
        └── lib.rs          # PyO3 bindings
```

## Core Classes

### Material

```python
@dataclass(frozen=True)
class Material:
    """Isotropic linear elastic material."""
    name: str
    E: float           # Young's modulus (Pa)
    nu: float          # Poisson's ratio
    rho: float = 0.0   # Density (kg/m³), optional

    @classmethod
    def steel(cls) -> Material:
        return cls("steel", E=200e9, nu=0.3, rho=7850)

    @classmethod
    def aluminum(cls) -> Material:
        return cls("aluminum", E=68.9e9, nu=0.33, rho=2700)
```

### Mesh

```python
class Mesh:
    """Immutable finite element mesh."""

    @classmethod
    def from_gmsh(cls, geometry, element_size: float) -> Mesh:
        """Generate mesh from Gmsh geometry."""
        ...

    @classmethod
    def from_file(cls, path: str, element_size: float = None) -> Mesh:
        """Load mesh from file (.msh, .step, .stl)."""
        ...

    @property
    def n_nodes(self) -> int: ...

    @property
    def n_elements(self) -> int: ...

    @property
    def bounds(self) -> BoundingBox: ...

    @property
    def element_types(self) -> set[ElementType]: ...
```

### Model

The Model class uses copy-on-write semantics:

```python
class Model:
    """Immutable FEA model with copy-on-write updates."""

    def __init__(
        self,
        mesh: Mesh,
        materials: dict[str, Material] = None,
    ):
        """Create a new model from mesh."""
        ...

    def assign(
        self,
        query: ElementQuery,
        material: str,
    ) -> Model:
        """Return new Model with material assigned to selected elements."""
        ...

    def constrain(
        self,
        query: NodeQuery,
        dofs: list[str],  # "ux", "uy", "uz", "rx", "ry", "rz"
        value: float = 0.0,
    ) -> Model:
        """Return new Model with displacement constraint."""
        ...

    def load(
        self,
        query: NodeQuery | FaceQuery,
        load: Load,
    ) -> Model:
        """Return new Model with applied load."""
        ...

    def define_component(
        self,
        name: str,
        query: Query,
    ) -> Model:
        """Return new Model with named component group."""
        ...

    def with_material(self, name: str, material: Material) -> Model:
        """Return new Model with additional material definition."""
        ...
```

### Load Types

```python
@dataclass(frozen=True)
class Force:
    """Concentrated force at nodes."""
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0

@dataclass(frozen=True)
class Pressure:
    """Surface pressure on faces."""
    value: float  # Pa, positive = into surface

@dataclass(frozen=True)
class Moment:
    """Concentrated moment at nodes."""
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0

Load = Force | Pressure | Moment
```

## Query DSL

### Node Queries

```python
class Nodes:
    """Query builder for node selection."""

    @classmethod
    def all(cls) -> NodeQuery:
        """Select all nodes."""
        ...

    @classmethod
    def where(
        cls,
        x: float | Callable = None,
        y: float | Callable = None,
        z: float | Callable = None,
        x__gt: float = None,
        x__lt: float = None,
        x__between: tuple[float, float] = None,
        # ... similar for y, z
        distance_to: tuple[Point | Line | Surface, float] = None,
    ) -> NodeQuery:
        """Select nodes matching predicates."""
        ...

class NodeQuery:
    """Lazy node selection, evaluated when needed."""

    def and_where(self, **kwargs) -> NodeQuery:
        """Intersect with additional predicate."""
        ...

    def or_where(self, **kwargs) -> NodeQuery:
        """Union with additional predicate."""
        ...

    def union(self, other: NodeQuery) -> NodeQuery:
        """Union of two queries."""
        ...

    def subtract(self, other: NodeQuery) -> NodeQuery:
        """Set difference."""
        ...

    def evaluate(self, mesh: Mesh) -> np.ndarray:
        """Evaluate query, return node indices."""
        ...
```

### Element Queries

```python
class Elements:
    """Query builder for element selection."""

    @classmethod
    def all(cls) -> ElementQuery: ...

    @classmethod
    def where(
        cls,
        type: ElementType = None,
        material: str = None,
        in_component: str = None,
        adjacent_to: NodeQuery = None,
    ) -> ElementQuery: ...
```

### Face Queries

```python
class Faces:
    """Query builder for surface face selection."""

    @classmethod
    def where(
        cls,
        normal: tuple[float, float, float] = None,
        on_boundary: bool = None,
        **spatial_kwargs,  # Same as Nodes.where
    ) -> FaceQuery: ...
```

## Solve Function

```python
def solve(
    model: Model,
    solver: SolverConfig = None,
) -> Results:
    """
    Solve the FEA problem.

    This is a pure function - it does not modify the model.

    Args:
        model: Complete model with mesh, materials, constraints, loads
        solver: Optional solver configuration

    Returns:
        Results object with displacements, stresses, etc.
    """
    ...
```

### Solver Configuration

```python
@dataclass
class SolverConfig:
    solver_type: Literal["auto", "direct", "iterative"] = "auto"
    auto_threshold: int = 100_000  # DOFs
    tolerance: float = 1e-10
    max_iterations: int = 1000

# Usage
results = solve(model, SolverConfig(solver_type="direct"))
```

## Results

```python
class Results:
    """Query-optimized access to solution data."""

    @property
    def displacement(self) -> DisplacementField:
        """Nodal displacement field."""
        ...

    @property
    def stress(self) -> StressField:
        """Element/integration point stress."""
        ...

    @property
    def strain(self) -> StrainField:
        """Element/integration point strain."""
        ...

    @property
    def reaction_force(self) -> ReactionField:
        """Reaction forces at constrained DOFs."""
        ...

    @property
    def von_mises(self) -> ScalarField:
        """Von Mises stress (derived, cached)."""
        ...

    @property
    def solver_stats(self) -> SolverStats:
        """Solver diagnostics."""
        ...

    def plot(
        self,
        field: str = "von_mises",
        **kwargs,
    ) -> None:
        """Render in Jupyter via PyVista."""
        ...

    def export(self, path: str) -> None:
        """Export to VTU for ParaView."""
        ...

    def save(self, path: str) -> None:
        """Save to HDF5 format."""
        ...

    @classmethod
    def load(cls, path: str) -> Results:
        """Load from HDF5."""
        ...
```

### Field Access with Queries

```python
class DisplacementField:
    def __getitem__(self, query: NodeQuery) -> np.ndarray:
        """Get displacements for selected nodes."""
        ...

    def magnitude(self) -> np.ndarray:
        """Displacement magnitude at all nodes."""
        ...

    def where(self, query: NodeQuery) -> DisplacementField:
        """Return filtered field."""
        ...

# Usage
top_disp = results.displacement[Nodes.where(z=mesh.bounds.z_max)]
max_disp = results.displacement.magnitude().max()
```

## Complete Example

```python
import mops
from mops import Model, Mesh, Material, Nodes, Elements, Faces
from mops import Force, Pressure, solve

# Define material
steel = Material.steel()

# Create mesh from CAD
mesh = Mesh.from_file("bracket.step", element_size=2.0)

# Build model (immutable, copy-on-write)
model = (
    Model(mesh, materials={"steel": steel})
    .assign(Elements.all(), material="steel")
    .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
    .load(
        Faces.where(normal=(1, 0, 0)).and_where(x__gt=90),
        Pressure(1e6)
    )
)

# Solve (pure function)
results = solve(model)

# Query results
print(f"Max displacement: {results.displacement.magnitude().max():.3e} m")
print(f"Max von Mises: {results.von_mises.max():.1f} MPa")

# Visualize
results.plot(field="von_mises")

# Export for ParaView
results.export("bracket_results.vtu")
```

## PyO3 Bindings

### Rust Side

```rust
use pyo3::prelude::*;

#[pyclass]
struct PyModel {
    inner: mops_core::Model,
}

#[pymethods]
impl PyModel {
    #[new]
    fn new(mesh: &PyMesh, materials: HashMap<String, PyMaterial>) -> Self { ... }

    fn assign(&self, query: &PyElementQuery, material: &str) -> PyModel { ... }
    fn constrain(&self, query: &PyNodeQuery, dofs: Vec<String>, value: f64) -> PyModel { ... }
    fn load(&self, query: &PyQuery, load: PyLoad) -> PyModel { ... }
}

#[pyfunction]
fn solve(model: &PyModel, config: Option<PySolverConfig>) -> PyResult<PyResults> {
    // Serialize Python objects to Rust structs
    // Call mops_core::solve()
    // Wrap results in Python objects
}
```

### Python Side (Type Stubs)

```python
# _core.pyi
class _Model:
    def __init__(self, mesh: _Mesh, materials: dict[str, _Material]) -> None: ...
    def assign(self, query: _ElementQuery, material: str) -> _Model: ...
    ...

def _solve(model: _Model, config: _SolverConfig | None) -> _Results: ...
```

## Error Handling

```python
class MopsError(Exception):
    """Base exception for MOPS errors."""
    pass

class MeshError(MopsError):
    """Invalid mesh (disconnected, degenerate elements)."""
    pass

class ModelError(MopsError):
    """Invalid model (missing materials, unconstrained)."""
    pass

class SolverError(MopsError):
    """Solver failure (singular matrix, non-convergent)."""
    pass

class QueryError(MopsError):
    """Invalid query (no matches, ambiguous)."""
    pass
```

## Testing Strategy

### Unit Tests

- Material property validation
- Query evaluation correctness
- Model copy-on-write semantics

### Integration Tests

- Full solve pipeline with known solutions
- Gmsh mesh import
- Results export/import roundtrip

### API Tests

- Type checking with mypy
- Docstring examples (doctest)

## Dependencies

```toml
[project]
dependencies = [
    "numpy>=1.20",
    "h5py>=3.0",
]

[project.optional-dependencies]
meshing = ["gmsh>=4.0"]
visualization = ["pyvista>=0.40"]
all = ["gmsh>=4.0", "pyvista>=0.40"]

[build-system]
requires = ["maturin>=1.0"]
build-backend = "maturin"
```

## Open Questions

1. **Async solve:** Worth providing `async def solve()` for Jupyter?
2. **Units:** Enforce SI or support pint/unyt?
3. **Serialization:** Pickle support for Model/Results?
4. **Progress callback:** How to report solver progress?
