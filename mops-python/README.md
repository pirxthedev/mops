# MOPS Python

Python bindings for the MOPS (Modular Open Physics Solver) finite element analysis library.

## Installation

### From Source (Development)

Requires Rust toolchain and maturin:

```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd mops-python
maturin develop
```

### Build Wheel

```bash
maturin build --release
pip install target/wheels/mops-*.whl
```

## Quick Start

```python
import numpy as np
from mops import Material, Mesh, solve_simple

# Define nodes for a simple tetrahedron
nodes = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
elements = np.array([[0, 1, 2, 3]], dtype=np.int64)

# Create mesh and material
mesh = Mesh(nodes, elements, "tet4")
steel = Material.steel()

# Define boundary conditions
constrained = np.array([0], dtype=np.int64)  # Fix node 0
loaded = np.array([3], dtype=np.int64)       # Load at node 3
force = np.array([0.0, 0.0, -1000.0])        # 1000N downward

# Solve
results = solve_simple(mesh, steel, constrained, loaded, force)
print(f"Max displacement: {results.max_displacement():.3e} m")
```

## Supported Element Types

- `tet4` - 4-node linear tetrahedron
- `tet10` - 10-node quadratic tetrahedron
- `hex8` - 8-node linear hexahedron

## API Reference

### Material

```python
# Create custom material
mat = Material("steel", e=200e9, nu=0.3, rho=7850)

# Preset materials
steel = Material.steel()
aluminum = Material.aluminum()
```

### Mesh

```python
# Create from numpy arrays
mesh = Mesh(nodes, elements, element_type)
# nodes: Nx3 array of coordinates
# elements: MxK array of node indices
# element_type: "tet4", "tet10", or "hex8"

mesh.n_nodes      # Number of nodes
mesh.n_elements   # Number of elements
```

### solve_simple

```python
results = solve_simple(
    mesh,                # Mesh object
    material,            # Material object
    constrained_nodes,   # Array of fixed node indices
    loaded_nodes,        # Array of loaded node indices
    load_vector,         # [fx, fy, fz] force vector
)
```

### Results

```python
results.displacement()           # Nx3 displacement array
results.displacement_magnitude() # N displacement magnitudes
results.max_displacement()       # Maximum displacement
```

## License

MIT OR Apache-2.0
