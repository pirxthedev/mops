# MOPS Examples

This directory contains runnable examples demonstrating the MOPS finite element analysis library.

## Prerequisites

Install MOPS from the project root:

```bash
cd mops
pip install -e ./mops-python
```

## Examples

### simple_tensile_bar.py

**Beginner** - A simple bar under uniaxial tension.

This is the simplest FEA problem and demonstrates:
- Creating a structured hex8 mesh
- Applying boundary conditions and loads
- Comparing results to analytical solutions

```bash
python examples/simple_tensile_bar.py
```

Expected output shows stress and elongation matching analytical values within 1%.

### cantilever_beam.py

**Intermediate** - Classic cantilever beam with tip load.

Demonstrates:
- Beam bending analysis
- Mesh convergence behavior
- Comparison with Euler-Bernoulli beam theory

```bash
python examples/cantilever_beam.py
```

The example shows tip deflection within 10-15% of analytical solution (varies with mesh density).

### plate_with_hole.py

**Advanced** - Stress concentration around a circular hole.

Demonstrates:
- Complex mesh generation (radial mesh around hole)
- Stress concentration analysis
- Von Mises stress extraction

```bash
python examples/plate_with_hole.py
```

Note: This example uses simplified boundary conditions (rigid hole), so Kt differs from the Kirsch solution. It demonstrates mesh generation and stress analysis workflow.

## Creating Your Own Models

Basic workflow:

```python
import numpy as np
from mops import (
    Elements, Force, Material, Mesh, Model, Nodes, solve
)

# 1. Create mesh (nodes and elements)
nodes = np.array([...], dtype=np.float64)  # (n_nodes, 3)
elements = np.array([...], dtype=np.int64)  # (n_elements, nodes_per_element)
mesh = Mesh(nodes, elements, "hex8")  # or "tet4", "tet10", "hex20"

# 2. Define material
steel = Material("steel", e=200e9, nu=0.3)

# 3. Build model (copy-on-write, immutable)
model = (
    Model(mesh, materials={"steel": steel})
    .assign(Elements.all(), material="steel")
    .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
    .load(Nodes.where(x=100), Force(fx=1000))
)

# 4. Solve
results = solve(model)

# 5. Extract results
disp = results.displacement()        # (n_nodes, 3) array
stress = results.stress()            # (n_elements, 6) Voigt notation
von_mises = results.von_mises()      # (n_elements,) scalar
max_disp = results.max_displacement()
```

## Query DSL

MOPS uses a query DSL for selecting nodes and elements:

```python
# Select nodes at x=0
fixed_nodes = Nodes.where(x=0)

# Select nodes with x between 10 and 20
mid_nodes = Nodes.where(x__gt=10, x__lt=20)

# Select all elements
all_elements = Elements.all()

# Combine queries
edge_nodes = Nodes.where(x=0) | Nodes.where(x=100)  # union
corner_nodes = Nodes.where(x=0) & Nodes.where(y=0)  # intersection

# Evaluate query to get indices
node_indices = fixed_nodes.evaluate(mesh)
```

## Element Types

MOPS supports these element types:

| Type | Nodes | Description |
|------|-------|-------------|
| `tet4` | 4 | Linear tetrahedron |
| `tet10` | 10 | Quadratic tetrahedron |
| `hex8` | 8 | Linear hexahedron |
| `hex20` | 20 | Quadratic hexahedron |

2D elements (plane stress, plane strain, axisymmetric) are available in mops-core but not yet exposed in the Python API.

## Tips

1. **Mesh density matters**: Coarse meshes are faster but less accurate
2. **Distribute loads**: When applying point loads, distribute across multiple nodes
3. **Check boundaries**: Verify constrained nodes have zero displacement
4. **Use von Mises**: For ductile materials, von Mises stress is the failure criterion
