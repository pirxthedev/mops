# HDF5 Results File Format Specification

**Date:** 2026-01-11
**Status:** Draft
**Depends on:** [Python API Design](2026-01-10-python-api-design.md)
**Related issue:** mops-ipq.1

## Overview

This specification defines the HDF5 file format for storing MOPS analysis results. The format is designed for:

1. **Large-scale results** - Efficient storage for meshes with millions of nodes
2. **Lazy access** - Load only what's needed via chunked datasets
3. **Query integration** - Fast subset extraction using indexed spatial data
4. **Self-contained** - Mesh, solution, and metadata in one file
5. **Reproducibility** - Store model definition for result provenance

### Design Principles

1. **Chunked storage** - All large arrays use HDF5 chunking for partial reads
2. **Compression** - GZIP level 4 for balance of speed and compression ratio
3. **Type safety** - Explicit dtypes with standardized conventions
4. **Versioned schema** - Format version for backwards compatibility
5. **Lazy evaluation** - Derived quantities computed on read, not stored

## File Extension

Files use the `.mops.h5` extension (e.g., `bracket_results.mops.h5`).

## File Structure

```
results.mops.h5
├── /metadata
│   ├── format_version          # string: "1.0"
│   ├── mops_version            # string: "0.1.0"
│   ├── created_at              # ISO 8601 timestamp
│   ├── analysis_type           # string: "linear_static"
│   └── description             # string: user-provided description
│
├── /mesh
│   ├── nodes                   # (n_nodes, 3) float64
│   ├── elements                # (n_elements, max_nodes) int64
│   ├── element_type            # string: "tet4", "tet10", "hex8", etc.
│   ├── element_offsets         # (n_elements,) int64 - for variable-length elements
│   └── bounds                  # (2, 3) float64 - [[x_min, y_min, z_min], [x_max, y_max, z_max]]
│
├── /materials
│   ├── names                   # (n_materials,) string
│   ├── properties              # (n_materials, 4) float64 - [E, nu, rho, alpha]
│   └── element_material_ids    # (n_elements,) int32 - index into names
│
├── /model
│   ├── /constraints
│   │   ├── node_indices        # (n_constrained,) int64
│   │   ├── dof_mask            # (n_constrained, 6) bool - [ux, uy, uz, rx, ry, rz]
│   │   └── prescribed_values   # (n_constrained, 6) float64 - non-zero for Dirichlet
│   │
│   └── /loads
│       ├── node_indices        # (n_loaded_nodes,) int64
│       ├── nodal_forces        # (n_loaded_nodes, 6) float64 - [Fx, Fy, Fz, Mx, My, Mz]
│       ├── face_elements       # (n_pressure_faces,) int64
│       ├── face_local_ids      # (n_pressure_faces,) int32
│       └── face_pressures      # (n_pressure_faces,) float64
│
├── /solution
│   ├── displacement            # (n_nodes, 3) float64 [ux, uy, uz]
│   ├── reaction_force          # (n_constrained_nodes, 3) float64
│   └── /solver
│       ├── type                # string: "cholesky", "pcg", "amg"
│       ├── iterations          # int32 (0 for direct solvers)
│       ├── residual_norm       # float64
│       ├── factorization_time  # float64 (seconds)
│       ├── solve_time          # float64 (seconds)
│       └── peak_memory_mb      # float64
│
├── /stress
│   ├── element                 # (n_elements, 6) float64 - Voigt notation
│   ├── element_von_mises       # (n_elements,) float64
│   ├── /nodal                  # Optional: nodal-averaged stresses
│   │   ├── tensor              # (n_nodes, 6) float64
│   │   └── von_mises           # (n_nodes,) float64
│   └── /integration_points     # Optional: full IP data
│       ├── stresses            # (total_ips, 6) float64
│       └── element_offsets     # (n_elements + 1,) int64 - CSR-like indexing
│
├── /strain                     # Optional: same structure as /stress
│   ├── element                 # (n_elements, 6) float64
│   └── element_von_mises       # (n_elements,) float64 - equivalent strain
│
└── /components                 # Named selections (like APDL CM)
    ├── names                   # (n_components,) string
    ├── types                   # (n_components,) string - "node", "element", "face"
    ├── offsets                 # (n_components + 1,) int64 - CSR indices
    └── indices                 # (total_indices,) int64 - packed indices
```

## Dataset Specifications

### Metadata Group (`/metadata`)

| Dataset | Type | Description |
|---------|------|-------------|
| `format_version` | string | Schema version, currently "1.0" |
| `mops_version` | string | MOPS version that created the file |
| `created_at` | string | ISO 8601 timestamp |
| `analysis_type` | string | "linear_static" (future: "nonlinear", "modal", etc.) |
| `description` | string | Optional user description |

### Mesh Group (`/mesh`)

| Dataset | Shape | Type | Chunking | Description |
|---------|-------|------|----------|-------------|
| `nodes` | (n_nodes, 3) | float64 | (1000, 3) | Node coordinates [x, y, z] |
| `elements` | (n_elements, max_nodes) | int64 | (1000, max_nodes) | Element connectivity |
| `element_type` | scalar | string | - | Element type identifier |
| `element_offsets` | (n_elements,) | int64 | (10000,) | For mixed meshes (optional) |
| `bounds` | (2, 3) | float64 | - | Axis-aligned bounding box |

**Element type identifiers:**
- `tet4` - 4-node tetrahedron
- `tet10` - 10-node tetrahedron
- `hex8` - 8-node hexahedron
- `hex20` - 20-node hexahedron
- `tri3` - 3-node triangle (2D)
- `quad4` - 4-node quadrilateral (2D)

**Connectivity convention:** Node indices are 0-based. For variable-length elements (mixed meshes), unused entries are padded with -1.

### Materials Group (`/materials`)

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `names` | (n_materials,) | string | Material identifiers |
| `properties` | (n_materials, 4) | float64 | [E, nu, rho, alpha] |
| `element_material_ids` | (n_elements,) | int32 | Material index per element |

**Property columns:**
- Column 0: `E` - Young's modulus (Pa)
- Column 1: `nu` - Poisson's ratio
- Column 2: `rho` - Density (kg/m³), 0 if not used
- Column 3: `alpha` - Thermal expansion coefficient (1/K), 0 if not used

### Model Group (`/model`)

#### Constraints Subgroup (`/model/constraints`)

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `node_indices` | (n_constrained,) | int64 | Constrained node indices |
| `dof_mask` | (n_constrained, 6) | bool | True = DOF constrained |
| `prescribed_values` | (n_constrained, 6) | float64 | Non-zero for prescribed displacement |

**DOF ordering:** [ux, uy, uz, rx, ry, rz] where r* are rotational DOFs (0 for 3D solids).

#### Loads Subgroup (`/model/loads`)

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `node_indices` | (n_loaded,) | int64 | Nodes with applied forces |
| `nodal_forces` | (n_loaded, 6) | float64 | [Fx, Fy, Fz, Mx, My, Mz] |
| `face_elements` | (n_faces,) | int64 | Element index for pressure faces |
| `face_local_ids` | (n_faces,) | int32 | Local face ID within element |
| `face_pressures` | (n_faces,) | float64 | Pressure magnitude (positive = inward) |

### Solution Group (`/solution`)

| Dataset | Shape | Type | Chunking | Description |
|---------|-------|------|----------|-------------|
| `displacement` | (n_nodes, 3) | float64 | (1000, 3) | Nodal displacements [ux, uy, uz] |
| `reaction_force` | (n_constrained, 3) | float64 | (1000, 3) | Reaction forces at constraints |

#### Solver Subgroup (`/solution/solver`)

| Dataset | Type | Description |
|---------|------|-------------|
| `type` | string | "cholesky", "pcg", "amg" |
| `iterations` | int32 | Iteration count (0 for direct) |
| `residual_norm` | float64 | Final residual norm |
| `factorization_time` | float64 | Factorization time (seconds) |
| `solve_time` | float64 | Solve time (seconds) |
| `peak_memory_mb` | float64 | Peak memory usage |

### Stress Group (`/stress`)

| Dataset | Shape | Type | Chunking | Description |
|---------|-------|------|----------|-------------|
| `element` | (n_elements, 6) | float64 | (1000, 6) | Element-averaged stress tensors |
| `element_von_mises` | (n_elements,) | float64 | (10000,) | Von Mises stress per element |

**Voigt notation ordering:** [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]

#### Optional Nodal Stress Subgroup (`/stress/nodal`)

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `tensor` | (n_nodes, 6) | float64 | Nodal-averaged stress tensors |
| `von_mises` | (n_nodes,) | float64 | Von Mises at nodes |

Nodal stresses are computed by averaging element stresses at shared nodes, weighted by element volume (optional feature).

#### Optional Integration Point Subgroup (`/stress/integration_points`)

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `stresses` | (total_ips, 6) | float64 | Stress at each integration point |
| `element_offsets` | (n_elements + 1,) | int64 | CSR-style offset array |

For element `i`, integration point stresses are at indices `element_offsets[i]:element_offsets[i+1]`.

### Components Group (`/components`)

Named selections stored in CSR (Compressed Sparse Row) format for efficient storage.

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `names` | (n_components,) | string | Component names |
| `types` | (n_components,) | string | Entity type: "node", "element", "face" |
| `offsets` | (n_components + 1,) | int64 | CSR offset array |
| `indices` | (total_indices,) | int64 | Packed index arrays |

For component `i`, indices are at `indices[offsets[i]:offsets[i+1]]`.

For face components, indices are stored as packed `(element_idx * 10 + local_face_idx)` values.

## Chunking Strategy

Chunk sizes are optimized for common access patterns:

| Data Type | Chunk Shape | Rationale |
|-----------|-------------|-----------|
| Node coordinates | (1000, 3) | Spatial queries read contiguous blocks |
| Element connectivity | (1000, max_nodes) | Element iteration |
| Displacements | (1000, 3) | Same as nodes |
| Element stress | (1000, 6) | Stress field visualization |
| Scalar fields | (10000,) | Quick statistics |

## Compression

All large datasets use GZIP compression with level 4 (shuffle filter enabled):

```python
dataset = f.create_dataset(
    "displacement",
    data=disp,
    chunks=(1000, 3),
    compression="gzip",
    compression_opts=4,
    shuffle=True,
)
```

## Query-Optimized Access

The format enables efficient query-based access:

### Spatial Queries

```python
# Read only bounding box first (tiny)
bounds = f["/mesh/bounds"][:]
x_min, x_max = bounds[0, 0], bounds[1, 0]

# Evaluate query to get indices (using mesh nodes)
nodes = f["/mesh/nodes"][:]  # or use spatial index if available
indices = Nodes.where(x=x_min).evaluate(mesh)

# Read only matching displacements (partial read)
disp = f["/solution/displacement"][indices, :]
```

### Component-Based Queries

```python
# Find component by name
names = [n.decode() for n in f["/components/names"][:]]
comp_idx = names.index("fixed_face")

# Get component indices (CSR extraction)
start, end = f["/components/offsets"][comp_idx:comp_idx+2]
node_indices = f["/components/indices"][start:end]

# Read subset
fixed_disp = f["/solution/displacement"][node_indices, :]
```

### Derived Quantities

Displacement magnitude is computed on-the-fly, not stored:

```python
# Read displacement
disp = f["/solution/displacement"][:]

# Compute magnitude
mag = np.linalg.norm(disp, axis=1)
```

This keeps file size smaller and avoids redundancy.

## Python API

### Results.save()

```python
def save(self, path: str, description: str = "") -> None:
    """Save results to HDF5 format.

    Args:
        path: Output file path (should end in .mops.h5)
        description: Optional description for metadata
    """
    import h5py

    with h5py.File(path, "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        meta.create_dataset("format_version", data="1.0")
        meta.create_dataset("mops_version", data=mops.__version__)
        meta.create_dataset("created_at", data=datetime.now().isoformat())
        meta.create_dataset("analysis_type", data="linear_static")
        meta.create_dataset("description", data=description)

        # Mesh
        mesh_grp = f.create_group("mesh")
        mesh_grp.create_dataset("nodes", data=self.mesh.coords,
                                 chunks=(1000, 3), compression="gzip")
        # ... etc
```

### Results.load()

```python
@classmethod
def load(cls, path: str) -> "Results":
    """Load results from HDF5 format.

    Args:
        path: Path to .mops.h5 file

    Returns:
        Results object with lazy-loaded data
    """
    import h5py

    f = h5py.File(path, "r")  # Keep open for lazy access

    # Validate format version
    version = f["/metadata/format_version"][()].decode()
    if version != "1.0":
        raise ValueError(f"Unsupported format version: {version}")

    return cls._from_hdf5(f)
```

## Versioning

The `format_version` field enables backwards compatibility:

- **1.0** - Initial format (this spec)
- Future versions may add groups/datasets but should remain readable by older code

Version compatibility matrix:

| File Version | MOPS Version | Notes |
|--------------|--------------|-------|
| 1.0 | 0.1.x | Initial release |

## File Size Estimates

For a 1 million node mesh:

| Data | Size (uncompressed) | Size (compressed ~3x) |
|------|--------------------|-----------------------|
| Nodes (1M × 3 × 8) | 24 MB | ~8 MB |
| Elements (500K × 4 × 8) | 16 MB | ~5 MB |
| Displacements | 24 MB | ~8 MB |
| Stress (500K × 6 × 8) | 24 MB | ~8 MB |
| **Total** | ~90 MB | **~30 MB** |

## Example File

A complete example with a simple tet4 mesh:

```python
import h5py
import numpy as np

with h5py.File("example.mops.h5", "w") as f:
    # Metadata
    meta = f.create_group("metadata")
    meta.create_dataset("format_version", data="1.0")
    meta.create_dataset("mops_version", data="0.1.0")
    meta.create_dataset("created_at", data="2026-01-11T12:00:00")
    meta.create_dataset("analysis_type", data="linear_static")
    meta.create_dataset("description", data="Cantilever beam")

    # Mesh (simple cantilever)
    mesh = f.create_group("mesh")
    nodes = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]
    ], dtype=np.float64)
    mesh.create_dataset("nodes", data=nodes)
    mesh.create_dataset("elements", data=np.array([[0, 1, 2, 3]]))
    mesh.create_dataset("element_type", data="tet4")
    mesh.create_dataset("bounds", data=np.array([
        [0, 0, 0], [1, 1, 1]
    ]))

    # Material
    mat = f.create_group("materials")
    mat.create_dataset("names", data=["steel"])
    mat.create_dataset("properties", data=np.array([[2.1e11, 0.3, 7850, 1.2e-5]]))
    mat.create_dataset("element_material_ids", data=np.array([0]))

    # Solution
    sol = f.create_group("solution")
    sol.create_dataset("displacement", data=np.array([
        [0, 0, 0], [0.001, -0.0002, 0], [0.0005, -0.0001, 0], [0.0005, -0.0001, 0.0003]
    ]))

    solver = sol.create_group("solver")
    solver.create_dataset("type", data="cholesky")
    solver.create_dataset("iterations", data=0)
    solver.create_dataset("residual_norm", data=1e-15)
    solver.create_dataset("factorization_time", data=0.001)
    solver.create_dataset("solve_time", data=0.0001)
    solver.create_dataset("peak_memory_mb", data=10.5)

    # Stress
    stress = f.create_group("stress")
    stress.create_dataset("element", data=np.array([
        [1e8, 0, 0, 0, 0, 0]  # Simple tension
    ]))
    stress.create_dataset("element_von_mises", data=np.array([1e8]))
```

## Validation

Files can be validated with the `mops validate` command:

```bash
mops validate results.mops.h5
```

Validation checks:
1. Format version is supported
2. Required groups/datasets exist
3. Array shapes are consistent
4. No NaN/Inf values in solution
5. Element indices are within bounds

## Future Extensions

The format is designed to accommodate future features:

- **Multiple load cases:** Add `/solution/case_001/`, `/solution/case_002/`
- **Time-dependent:** Add `/solution/step_001/` with time attribute
- **Modal analysis:** Add `/modes/` group with frequencies and mode shapes
- **Adaptive mesh:** Add `/mesh/refinement_levels/`

## References

- [HDF5 Documentation](https://portal.hdfgroup.org/display/HDF5/HDF5)
- [h5py User Guide](https://docs.h5py.org/en/stable/)
- [ANSYS Result File Formats](https://ansyshelp.ansys.com/)
- [CGNS Standard](https://cgns.github.io/) - Similar hierarchical approach
