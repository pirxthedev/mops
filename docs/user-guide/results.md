# Working with Results

The `Results` class provides access to analysis outputs including displacements, stresses, and derived quantities. Results can be persisted to HDF5 format for later analysis.

## Basic Access

After solving, access results through the `Results` object:

```python
from mops import solve

results = solve(model)

# Displacements: (n_nodes, 3) array
disp = results.displacement()

# Stresses: (n_elements, 6) array in Voigt notation
stress = results.stress()

# Von Mises stress: (n_elements,) scalar per element
von_mises = results.von_mises()
```

## Displacement Data

Displacement is stored per node with three components (ux, uy, uz):

```python
disp = results.displacement()  # Shape: (n_nodes, 3)

# Access individual components
ux = disp[:, 0]  # x-displacement
uy = disp[:, 1]  # y-displacement
uz = disp[:, 2]  # z-displacement

# Maximum displacement magnitude
max_disp = results.max_displacement()

# Node displacement
node_disp = results.node_displacement(node_id=42)  # Returns (3,) array
```

## Stress Data

Stress is stored per element in Voigt notation:

```python
stress = results.stress()  # Shape: (n_elements, 6)

# Component order: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
sigma_xx = stress[:, 0]
sigma_yy = stress[:, 1]
sigma_zz = stress[:, 2]
tau_xy = stress[:, 3]
tau_yz = stress[:, 4]
tau_xz = stress[:, 5]

# Stress at specific element
elem_stress = results.element_stress(element_id=10)  # Returns (6,) array
```

## Derived Quantities

Results provides several derived stress quantities:

### Von Mises (Equivalent) Stress

The von Mises stress is the most common failure criterion for ductile materials:

```python
von_mises = results.von_mises()
max_vm = results.max_von_mises()

# Check against yield strength
if max_vm > yield_strength:
    print("Warning: Yielding may occur")
```

### Principal Stresses

Eigenvalues of the stress tensor, sorted as σ₁ ≥ σ₂ ≥ σ₃:

```python
principals = results.principal_stresses()  # Shape: (n_elements, 3)

sigma_1 = principals[:, 0]  # Maximum principal
sigma_2 = principals[:, 1]  # Intermediate principal
sigma_3 = principals[:, 2]  # Minimum principal
```

### Tresca Stress

Maximum shear stress criterion (more conservative than von Mises):

```python
tresca = results.tresca()  # σ₁ - σ₃
max_tresca = results.max_tresca()
```

### Additional Quantities

```python
# Maximum shear stress: τ_max = (σ₁ - σ₃) / 2
max_shear = results.max_shear_stress()

# Hydrostatic stress: σ_h = (σ_xx + σ_yy + σ_zz) / 3
hydro = results.hydrostatic_stress()

# Pressure: p = -σ_h
pressure = results.pressure()

# Stress intensity: max(|σ₁ - σ₂|, |σ₂ - σ₃|, |σ₃ - σ₁|)
intensity = results.stress_intensity()
```

## Query-Optimized Fields

For large results or HDF5-backed data, use field accessors with queries:

```python
# Get displacement at specific nodes
from mops import Nodes

tip_nodes = Nodes.where(x=100)
tip_disp = results.displacement_field[tip_nodes]

# Von Mises at elements near a region
from mops import Elements

region_elements = Elements.touching(Nodes.in_sphere((50, 50, 50), radius=10))
region_vm = results.von_mises_field[region_elements]

# Lazy filtering (efficient for HDF5)
filtered = results.displacement_field.where(Nodes.where(x=0))
data = filtered.values()  # Data only loaded now
```

### Displacement Magnitude

```python
mag_field = results.displacement_field.magnitude()
max_mag = mag_field.max()
min_mag = mag_field.min()
avg_mag = mag_field.mean()
```

## Solver Statistics

Access solver performance metrics:

```python
stats = results.solve_stats

if stats:
    print(f"Solver: {stats['solver']}")
    print(f"Total time: {stats['total_time']:.3f} s")
    print(f"DOFs: {stats['n_dofs']}")
    print(f"Non-zeros: {stats['n_nonzeros']}")

    # For iterative solvers
    if stats['iterations'] > 0:
        print(f"Iterations: {stats['iterations']}")
        print(f"Final residual: {stats['residual']:.2e}")
```

## HDF5 Persistence

### Saving Results

Save results to HDF5 format for later analysis:

```python
# Basic save
results.save("analysis.mops.h5")

# Include model for query support after loading
results.save("analysis.mops.h5", model=model, description="Bracket stress analysis")
```

The saved file contains:
- `/metadata`: Version, timestamp, description
- `/mesh`: Nodes, elements, bounds (if model provided)
- `/materials`: Material definitions and assignments (if model provided)
- `/solution`: Displacements, reaction forces
- `/stress`: Element stresses, von Mises, derived quantities
- `/model/constraints`: Boundary conditions (if model provided)
- `/model/loads`: Applied loads (if model provided)

### Loading Results

```python
# Load with lazy reading (file stays open)
loaded = Results.load("analysis.mops.h5")

# Access data (reads from disk on demand)
disp = loaded.displacement()
vm = loaded.von_mises()

# Query support (requires mesh data in file)
tip_disp = loaded.displacement_field[Nodes.where(x=100)]

# Always close when done
loaded.close()

# Or use context manager
with Results.load("analysis.mops.h5") as results:
    print(f"Max von Mises: {results.max_von_mises():.2f} MPa")
```

### Lazy Loading Benefits

HDF5-backed results use lazy loading:
- File opens but data isn't read into memory
- Only accessed data is loaded
- Query-filtered reads only load needed subset
- Efficient for large models (millions of elements)

```python
# Only reads subset of displacement data
with Results.load("large_analysis.mops.h5") as results:
    # These reads are efficient - only load needed data
    tip_disp = results.displacement_field[Nodes.where(x__gt=90)]
    max_vm = results.max_von_mises()  # Reads von Mises array only
```

## File Format Details

The HDF5 file uses:
- **Chunking**: Large arrays stored in 1000-row chunks for partial reads
- **GZIP compression**: ~3x size reduction with shuffle filter
- **Format version**: `1.0` (stored in `/metadata/format_version`)

Typical file sizes:
- 100k node mesh: ~3 MB compressed
- 1M node mesh: ~30 MB compressed

## Practical Examples

### Finding Critical Locations

```python
import numpy as np

# Find element with maximum stress
vm = results.von_mises()
critical_elem = np.argmax(vm)
print(f"Critical element: {critical_elem}")
print(f"Max von Mises: {vm[critical_elem]:.1f} MPa")

# Get stress tensor at critical element
stress = results.element_stress(critical_elem)
print(f"Stress components: {stress}")
```

### Post-Processing with NumPy

```python
import numpy as np

# Statistics
disp = results.displacement()
print(f"Mean displacement: {np.mean(disp, axis=0)}")
print(f"Std displacement: {np.std(disp, axis=0)}")

# Find nodes exceeding displacement limit
max_allowed = 1.0  # mm
disp_mag = np.linalg.norm(disp, axis=1)
exceed = np.where(disp_mag > max_allowed)[0]
print(f"Nodes exceeding limit: {len(exceed)}")
```

### Comparison Across Analyses

```python
# Compare two load cases
with Results.load("case1.mops.h5") as r1, Results.load("case2.mops.h5") as r2:
    vm1 = r1.max_von_mises()
    vm2 = r2.max_von_mises()
    print(f"Case 1 max VM: {vm1:.1f} MPa")
    print(f"Case 2 max VM: {vm2:.1f} MPa")
    print(f"Increase: {(vm2 - vm1) / vm1 * 100:.1f}%")
```

## API Reference

### Core Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `displacement()` | (n_nodes, 3) | Nodal displacement vectors |
| `stress()` | (n_elements, 6) | Element stress tensors (Voigt) |
| `von_mises()` | (n_elements,) | Von Mises equivalent stress |
| `max_displacement()` | float | Maximum displacement magnitude |
| `max_von_mises()` | float | Maximum von Mises stress |

### Derived Quantities

| Method | Returns | Description |
|--------|---------|-------------|
| `principal_stresses()` | (n_elements, 3) | Principal stresses [σ₁, σ₂, σ₃] |
| `tresca()` | (n_elements,) | Tresca stress (σ₁ - σ₃) |
| `max_shear_stress()` | (n_elements,) | Maximum shear (τ_max = (σ₁ - σ₃)/2) |
| `hydrostatic_stress()` | (n_elements,) | Mean stress ((σ_xx + σ_yy + σ_zz)/3) |
| `pressure()` | (n_elements,) | Pressure (-σ_hydrostatic) |
| `stress_intensity()` | (n_elements,) | Stress intensity |

### Field Accessors

| Property | Type | Query Support |
|----------|------|---------------|
| `displacement_field` | NodeField | Yes |
| `stress_field` | ElementField | Yes |
| `von_mises_field` | ScalarElementField | Yes |
| `tresca_field` | ScalarElementField | Yes |
| `hydrostatic_field` | ScalarElementField | Yes |

### Persistence

| Method | Description |
|--------|-------------|
| `save(path, model=None, description="")` | Save to HDF5 |
| `Results.load(path)` | Load from HDF5 (lazy) |
| `close()` | Close HDF5 file handle |

## Next Steps

- [Visualization](visualization.md): Plot stress fields and deformations
- [Query DSL](queries.md): Advanced result filtering
- [Elements](elements.md): Understanding stress output by element type
