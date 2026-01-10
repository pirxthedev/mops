# MOPS Initial Design

**Date:** 2026-01-10
**Status:** Approved

## Overview

MOPS (Modular Open Physics Solver) is an open source finite element analysis tool designed to replace ANSYS for structural mechanics simulations. It combines a high-performance Rust solver core with a Pythonic declarative API.

## Goals

- Target industry engineers with reliability, validation, and certification-ready results
- Emphasize modern programming principles: functional programming, immutable data structures
- Provide APDL-like selection DSL using declarative, SQL-inspired queries
- Handle large-scale industrial models (10+ GB result files)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Python Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Model API   │  │ Mesh API    │  │ Results/Viz     │ │
│  │ (immutable) │  │ (Gmsh wrap) │  │ (PyVista)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  PyO3 Boundary                          │
│         (serialize Model+Mesh → Rust structs)           │
├─────────────────────────────────────────────────────────┤
│                     Rust Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Assembly    │  │ Solver      │  │ Element Library │ │
│  │ (Rayon)     │  │ (SuiteSparse│  │ (2D/3D kernels) │ │
│  │             │  │  + hypre)   │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Key Design Principles

- **Immutability**: Python `Model` and `Mesh` are frozen. No mutation after construction.
- **Pure functions**: `solve(model, mesh) → Results` is a pure transformation.
- **Separation of concerns**: Python owns the "what", Rust owns the "how fast".

## Scope

### Initial Target

- **Analysis type:** Linear static
- **Elements:** 2D (plane stress, plane strain, axisymmetric) + 3D (tet4, tet10, hex8)
- **Parallelism:** Shared-memory via Rayon

### Future Extensions

- Nonlinear static (large deformation, plasticity, contact)
- Linear dynamics (modal, harmonic, transient)
- MPI for distributed memory (HPC clusters)
- GPU acceleration

## Query-Based Selection System

Selections are lazy query objects - declarative descriptions evaluated when needed:

```python
# Queries are composable, lazy, and reusable
left_face = Nodes.where(x=0)
right_face = Nodes.where(x=L)
hole_rim = Nodes.where(distance_to(Point(5, 5, 0)) < 1.01 * radius)

# Logical composition
fixed_nodes = left_face.union(bolt_holes).subtract(pilot_hole)

# Copy-on-write model construction
model = (
    Model(mesh, materials={"steel": steel})
    .assign(Elements.all(), material="steel")
    .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
    .load(Faces.where(normal=(1,0,0)).and_where(x=L), Pressure(1e6))
)

# Named component groups (like APDL CM command)
model = model.define_component("LEFT_FACE", Nodes.where(x=0))
```

### Query Predicates

- **Spatial:** `x=`, `y__gt=`, `z__between=(a,b)`, `distance_to(point/line/surface)`
- **Topological:** `adjacent_to=`, `connected_to=`, `on_surface=`
- **Logical:** `.and_where()`, `.or_where()`, `.subtract()`, `.union()`
- **Lambdas:** `where(lambda n: custom_condition(n))`

## Rust Solver Core

### Element Trait

```rust
pub trait Element: Send + Sync {
    fn dofs_per_node(&self) -> usize;

    fn stiffness(
        &self,
        coords: &[Point3],
        material: &Material,
    ) -> DMatrix<f64>;

    fn stress(
        &self,
        coords: &[Point3],
        displacements: &[f64],
        material: &Material,
    ) -> Vec<StressTensor>;
}
```

### Parallel Assembly

```rust
pub fn assemble(mesh: &Mesh, model: &Model) -> CsrMatrix<f64> {
    let element_matrices: Vec<_> = mesh.elements
        .par_iter()  // Rayon parallel iterator
        .map(|elem| {
            let coords = mesh.element_coords(elem);
            let mat = model.material_for(elem);
            (elem.connectivity(), elem.stiffness(&coords, mat))
        })
        .collect();

    assemble_global(mesh.n_dofs(), element_matrices)
}
```

### Linear Solvers

- **Direct:** SuiteSparse (Cholesky/LU) for small-medium problems
- **Iterative:** hypre BoomerAMG for large problems
- **Auto-selection:** Based on problem size with explicit override option

## Results File Format

HDF5-based format for large-scale results with lazy, query-optimized access.

### File Structure

```
results.mops.h5
├── /mesh
│   ├── nodes                   # (n_nodes, 3) float64
│   ├── elements/
│   └── spatial_index/          # For query acceleration
│
├── /solution
│   ├── displacement            # (n_nodes, 3) float64, chunked
│   ├── reaction_force
│   └── solver_stats
│
├── /derived                    # Computed on-demand, cached
│   ├── stress/
│   ├── strain/
│   └── von_mises
│
├── /components                 # Named selections preserved
│
└── /metadata
```

### Query-Optimized Access

Queries are evaluated against mesh metadata first, yielding indices before touching large result arrays:

```python
# Query evaluated against mesh (small), not results (huge)
top_surface = Nodes.where(z=lambda m: m.bounds.z_max)

# Only reads/computes for matching nodes
max_vm = results.von_mises.where(top_surface).max()
```

## Meshing

Gmsh integration for industrial-strength meshing:

```python
mesh = Mesh.from_gmsh(geometry, element_size=0.01)
mesh = Mesh.from_file("part.msh")
mesh = Mesh.from_file("part.step", element_size=0.5)  # CAD import
```

## Visualization

Jupyter integration via PyVista:

```python
results.plot(field="von_mises")  # renders in notebook
results.export("output.vtu")     # for ParaView
```

## Verification Strategy

### Three-Tier Testing

1. **Unit tests** (fast, every commit): Element stiffness, assembly, query DSL
2. **Integration tests** (minutes, every PR): Full pipeline, Gmsh import
3. **Verification tests** (hours, nightly): Analytical benchmarks, NAFEMS suite

### Benchmarks

- **Analytical:** Cantilever beam, plate with hole, pressurized cylinder
- **NAFEMS:** LE1, LE10, and other standard benchmarks

## Repository Structure

```
mops/
├── mops-core/                   # Rust solver library
│   ├── Cargo.toml
│   └── src/
│
├── mops-python/                 # Python bindings + API
│   ├── pyproject.toml
│   ├── src/mops/
│   └── rust/                    # PyO3 bindings
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── verification/
│
├── docs/
│   ├── plans/
│   ├── user-guide/
│   └── theory/
│
└── examples/
```

## Research Tasks Required

1. **APDL Selection DSL:** Reverse-engineer selection commands (NSEL, ESEL, KSEL, LSEL, ASEL, VSEL, CM, CMSEL) from ANSYS documentation to inform query DSL design
2. **NAFEMS Benchmarks:** Acquire and document benchmark specifications (geometry, loads, materials, reference values)

## Dependencies

### Rust

- `rayon` - Parallel iteration
- `nalgebra` / `ndarray` - Linear algebra
- `sprs` - Sparse matrices
- `suitesparse-sys` - Direct solver bindings
- `hdf5` - Result file I/O
- `pyo3` - Python bindings

### Python

- `numpy` - Array operations
- `h5py` - HDF5 access
- `gmsh` - Meshing
- `pyvista` - Visualization
- `maturin` - Build tooling

## Open Questions

1. **License:** MIT vs Apache-2.0?
2. **Units:** Enforce SI internally, or support unit systems?
3. **GPU:** Worth targeting initially, or defer?
