# Getting Started

This guide walks through your first finite element analysis with MOPS. We'll analyze a simple tensile bar and compare results to the analytical solution.

## The Problem

A rectangular aluminum bar is pulled in tension:

```
         ← Fixed                     Force →

    ┌────────────────────────────────────────┐
    │                                        │ ← Bar (100 x 10 x 10 mm)
    └────────────────────────────────────────┘
    x=0                                   x=100
```

- **Geometry**: 100 mm long, 10 x 10 mm cross-section
- **Material**: Aluminum (E = 70 GPa, ν = 0.33)
- **Boundary conditions**: Fixed at x=0, 10 kN force at x=100

### Analytical Solution

For uniaxial tension:

```
Stress:      σ = F/A = 10,000 N / 100 mm² = 100 MPa
Strain:      ε = σ/E = 100 / 70,000 = 0.00143
Elongation:  δ = ε × L = 0.00143 × 100 = 0.143 mm
```

## Step 1: Import MOPS

```python
import numpy as np
from mops import (
    Mesh,
    Model,
    Material,
    Nodes,
    Elements,
    Force,
    solve,
)
```

## Step 2: Create the Mesh

MOPS works with node coordinates and element connectivity. For structured geometries, you can generate meshes programmatically:

```python
def create_bar_mesh(length, width, height, nx, ny, nz):
    """Create hex8 mesh for a rectangular bar."""
    # Node coordinates
    x = np.linspace(0, length, nx + 1)
    y = np.linspace(0, width, ny + 1)
    z = np.linspace(0, height, nz + 1)

    nodes = []
    for zi in z:
        for yi in y:
            for xi in x:
                nodes.append([xi, yi, zi])
    nodes = np.array(nodes, dtype=np.float64)

    # Element connectivity (hex8)
    def idx(i, j, k):
        return i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append([
                    idx(i, j, k),         idx(i+1, j, k),
                    idx(i+1, j+1, k),     idx(i, j+1, k),
                    idx(i, j, k+1),       idx(i+1, j, k+1),
                    idx(i+1, j+1, k+1),   idx(i, j+1, k+1),
                ])
    elements = np.array(elements, dtype=np.int64)

    return Mesh(nodes, elements, "hex8")

mesh = create_bar_mesh(100, 10, 10, nx=10, ny=2, nz=2)
print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
```

Output:
```
Mesh: 99 nodes, 40 elements
```

## Step 3: Define Material

```python
aluminum = Material("aluminum", e=70e3, nu=0.33)  # E in MPa
```

Common materials can also be created with predefined values:
```python
steel = Material.steel()      # E=200 GPa, nu=0.3
aluminum = Material.aluminum() # E=70 GPa, nu=0.33
```

## Step 4: Build the Model

MOPS uses immutable, copy-on-write semantics. Each method returns a new model:

```python
# Count nodes at loaded end to distribute force
loaded_nodes = Nodes.where(x=100).evaluate(mesh)
force_per_node = 10000 / len(loaded_nodes)  # 10 kN total

model = (
    Model(mesh, materials={"aluminum": aluminum})
    .assign(Elements.all(), material="aluminum")
    .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
    .load(Nodes.where(x=100), Force(fx=force_per_node))
)
```

Key concepts:
- `assign()`: Links elements to materials
- `constrain()`: Applies displacement boundary conditions
- `load()`: Applies forces or pressures

## Step 5: Solve

```python
results = solve(model)
```

The solver automatically selects the best algorithm:
- **Direct (Cholesky)**: For problems < 100k DOFs
- **Iterative (PCG+AMG)**: For larger problems

## Step 6: Extract Results

```python
# Displacement array: (n_nodes, 3) - [ux, uy, uz] per node
disp = results.displacement()

# Stress array: (n_elements, 6) - Voigt notation [xx, yy, zz, xy, yz, xz]
stress = results.stress()

# Von Mises stress: (n_elements,) - scalar per element
von_mises = results.von_mises()

# Summary statistics
print(f"Max displacement: {results.max_displacement():.6e} mm")
print(f"Max von Mises: {np.max(von_mises):.1f} MPa")
```

Output:
```
Max displacement: 1.43e-01 mm
Max von Mises: 100.0 MPa
```

## Step 7: Verify Results

Compare to analytical solution:

```python
# Get x-displacement at loaded end
avg_elongation = np.mean([disp[n, 0] for n in loaded_nodes])
print(f"FEA elongation: {avg_elongation:.4f} mm")
print(f"Analytical:     {0.143:.4f} mm")
print(f"Error: {abs(avg_elongation - 0.143) / 0.143 * 100:.2f}%")

# Check stress
avg_stress_xx = np.mean(stress[:, 0])  # sigma_xx
print(f"FEA stress: {avg_stress_xx:.1f} MPa")
print(f"Analytical: 100.0 MPa")
```

## Complete Example

```python
#!/usr/bin/env python3
"""Simple tensile bar analysis."""

import numpy as np
from mops import Mesh, Model, Material, Nodes, Elements, Force, solve

# Create mesh
def create_bar_mesh(L, W, H, nx, ny, nz):
    x = np.linspace(0, L, nx + 1)
    y = np.linspace(0, W, ny + 1)
    z = np.linspace(0, H, nz + 1)

    nodes = np.array([[xi, yi, zi]
                      for zi in z for yi in y for xi in x], dtype=np.float64)

    def idx(i, j, k):
        return i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    elements = np.array([
        [idx(i, j, k), idx(i+1, j, k), idx(i+1, j+1, k), idx(i, j+1, k),
         idx(i, j, k+1), idx(i+1, j, k+1), idx(i+1, j+1, k+1), idx(i, j+1, k+1)]
        for k in range(nz) for j in range(ny) for i in range(nx)
    ], dtype=np.int64)

    return Mesh(nodes, elements, "hex8")

mesh = create_bar_mesh(100, 10, 10, nx=10, ny=2, nz=2)

# Material and model
aluminum = Material("aluminum", e=70e3, nu=0.33)
loaded_nodes = Nodes.where(x=100).evaluate(mesh)
force_per_node = 10000 / len(loaded_nodes)

model = (
    Model(mesh, materials={"aluminum": aluminum})
    .assign(Elements.all(), material="aluminum")
    .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
    .load(Nodes.where(x=100), Force(fx=force_per_node))
)

# Solve and report
results = solve(model)
print(f"Max displacement: {results.max_displacement():.4e} mm")
print(f"Max von Mises: {np.max(results.von_mises()):.1f} MPa")
```

## Next Steps

- [Query DSL](queries.md): Advanced node/element selection
- [Elements](elements.md): Element types and formulations
- [Results](results.md): Working with analysis results
- [Visualization](visualization.md): Plotting stress fields

See also the [examples directory](../../examples/) for more complete examples:
- `simple_tensile_bar.py`: Basic stress/strain validation
- `cantilever_beam.py`: Beam bending with tip load
- `plate_with_hole.py`: Stress concentration analysis
