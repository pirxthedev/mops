# MOPS — Modular Open Physics Solver

A high-performance finite element analysis (FEA) solver written in Rust with a Pythonic declarative API. Designed as an open-source alternative to commercial FEA tools.

## Features

- **Rust Core**: Parallel assembly and solving via Rayon, sparse Cholesky (faer) and iterative PCG+AMG (kryst) solvers
- **Full Element Library**: Tet4, Tet10, Hex8, Hex20, plane stress/strain, axisymmetric elements
- **Query DSL**: Select nodes, elements, and faces with spatial predicates and set operations
- **HDF5 Results**: Lazy-loading results with query-optimized access
- **PyVista Integration**: In-notebook visualization of stress fields
- **NAFEMS Verified**: Benchmarked against LE1, LE10, and analytical solutions

## Installation

### Prerequisites

- Rust 1.75+ (for building mops-core)
- Python 3.10+
- Node.js (for beads issue tracker)

### Build from Source

```bash
# Build Rust core
cd mops-core
cargo build --release

# Install Python package
cd ../mops-python
pip install -e .
```

## Quick Start

```python
from mops import Model, Nodes, Elements, Faces, solve
from mops import Steel, Force, Pressure

# Load mesh (Gmsh format)
mesh = Mesh.load("cantilever.msh")

# Build model with immutable, copy-on-write semantics
model = (Model(mesh)
    .assign(Elements.all(), material=Steel)
    .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
    .load(Faces.where(x=100), Pressure(1e6)))

# Solve
results = solve(model)

# Access results
displacement = results.displacement
von_mises = results.von_mises_stress

# Visualize
results.plot(field="von_mises")

# Save to HDF5
results.save("output.mops.h5")
```

## Query DSL

Select mesh entities using spatial predicates:

```python
# Node selection
fixed_nodes = Nodes.where(x=0)                    # x == 0
top_nodes = Nodes.where(y__gt=10)                 # y > 10
interior = Nodes.where(z__between=(5, 15))        # 5 <= z <= 15

# Element selection
all_elements = Elements.all()
tet_elements = Elements.where(type="tet4")

# Face selection (for pressure loads)
pressure_face = Faces.where(normal=(1, 0, 0))
boundary = Faces.on_boundary()

# Set operations
combined = fixed_nodes | top_nodes               # union
refined = all_elements & Elements.where(x__lt=50) # intersection
excluded = all_elements - tet_elements            # difference
inverted = ~fixed_nodes                           # invert

# Named components (like APDL CM command)
model = model.define_component("support", Nodes.where(x=0))
```

## Element Library

| Element | Type | Nodes | Integration | Use Case |
|---------|------|-------|-------------|----------|
| Tet4 | 3D | 4 | 1-point | General meshing |
| Tet10 | 3D | 10 | 4-point | Higher accuracy |
| Hex8 | 3D | 8 | 2×2×2 | Structured meshes |
| Hex20 | 3D | 20 | 3×3×3 | High accuracy |
| Tri3/Quad4 | 2D | 3-4 | varies | Plane stress/strain |
| Axisymmetric | 2D | 3-4 | varies | Rotational symmetry |

## Project Structure

```
mops/
├── mops-core/           # Rust solver core
│   └── src/
│       ├── element/     # Element implementations
│       ├── solver.rs    # Cholesky & iterative solvers
│       ├── assembly.rs  # Parallel FE assembly
│       └── stress.rs    # Stress recovery
├── mops-python/         # Python API
│   └── src/mops/
│       ├── model.py     # Model builder
│       ├── query.py     # Query DSL
│       ├── results.py   # Results + HDF5 I/O
│       └── viz.py       # PyVista integration
└── docs/
    └── plans/           # Architecture specifications
```

## Development

### Running Tests

```bash
# Rust tests
cd mops-core && cargo test

# Python tests
cd mops-python && pytest
```

### Issue Tracking

This project uses [Beads](https://github.com/steveyegge/beads) for git-backed issue tracking:

```bash
npm install -g @beads/bd
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress
bd close <id>
bd sync
```

See [AGENTS.md](./AGENTS.md) for agent workflow instructions.

## Architecture

MOPS uses a layered architecture:

1. **Python Layer**: Declarative model building with immutable, copy-on-write semantics
2. **PyO3 Boundary**: Serialization bridge between Python and Rust
3. **Rust Core**: High-performance assembly, solving, and stress recovery

Solvers auto-select based on problem size:
- **Direct (Cholesky)**: <100k DOFs, uses faer sparse solver
- **Iterative (PCG+AMG)**: ≥100k DOFs, uses kryst solver (optional feature)

## Documentation

- [Architecture Design](docs/plans/2026-01-10-mops-initial-design.md)
- [Element Library](docs/plans/2026-01-10-element-library-design.md)
- [Query DSL Specification](docs/plans/2026-01-11-query-dsl-spec.md)
- [HDF5 Results Format](docs/plans/2026-01-11-hdf5-results-format-spec.md)
- [NAFEMS Benchmarks](docs/plans/2026-01-11-nafems-benchmarks-spec.md)

## License

MIT
