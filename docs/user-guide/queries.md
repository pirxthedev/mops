# Query DSL

The Query DSL provides a declarative way to select nodes, elements, and faces from a mesh. It's inspired by ANSYS APDL's selection commands but with a Pythonic interface.

## Overview

Queries are **lazy** - they store predicates but don't execute until evaluated. This allows building complex selections through composition.

```python
from mops import Nodes, Elements, Faces

# Create a query (nothing evaluated yet)
fixed_nodes = Nodes.where(x=0)

# Evaluate against a mesh to get indices
indices = fixed_nodes.evaluate(mesh)  # Returns numpy array of node indices
```

## Node Queries

### Basic Selection

```python
# All nodes
Nodes.all()

# By coordinate (exact match with tolerance)
Nodes.where(x=0)           # x == 0
Nodes.where(x=0, y=0)      # x == 0 AND y == 0

# By index
Nodes.by_indices([0, 1, 2, 3])
```

### Comparison Operators

Use double-underscore syntax for comparisons:

```python
# Greater than / Less than
Nodes.where(x__gt=10)      # x > 10
Nodes.where(x__lt=50)      # x < 50
Nodes.where(x__gte=0)      # x >= 0
Nodes.where(x__lte=100)    # x <= 100

# Range (inclusive)
Nodes.where(x__between=(10, 50))    # 10 <= x <= 50

# Combine predicates (AND)
Nodes.where(x__gt=0, y__lt=100)     # x > 0 AND y < 100
```

### Spatial Queries

```python
# Near a point (within tolerance)
Nodes.near_point((50, 25, 0), tol=0.5)

# Inside a sphere
Nodes.in_sphere(center=(50, 50, 50), radius=10)

# Inside an axis-aligned box
Nodes.in_box(
    min_corner=(0, 0, 0),
    max_corner=(100, 50, 25)
)

# Near a line segment
Nodes.near_line(
    start=(0, 0, 0),
    end=(100, 0, 0),
    tol=0.5
)
```

### Topological Queries

```python
# Nodes on selected elements (like APDL NSLE)
Nodes.on_elements(Elements.all())
Nodes.on_elements(Elements.where(material="steel"))

# Combine with spatial queries
Nodes.on_elements(Elements.touching(Nodes.where(x=0)))
```

## Element Queries

### Basic Selection

```python
# All elements
Elements.all()

# By index
Elements.by_indices([0, 1, 2])
```

### Topological Queries

```python
# Elements where ALL nodes are in selection (like APDL ESLN)
Elements.attached_to(Nodes.where(x=0))

# Elements where ANY node is in selection
Elements.touching(Nodes.where(x=0))
```

The difference:
- `attached_to`: Element is selected only if **every** node is in the selection
- `touching`: Element is selected if **any** node is in the selection

```
Elements at x=0 boundary:

        attached_to              touching
        ┌──┬──┬──┐              ┌──┬──┬──┐
        │  │  │  │              │██│  │  │
x=0 →   │  │  │  │      x=0 →   │██│  │  │
        │  │  │  │              │██│  │  │
        └──┴──┴──┘              └──┴──┴──┘
        (none selected)         (first column selected)
```

## Face Queries

Faces are identified as (element_index, local_face_index) pairs.

### Basic Selection

```python
# Faces on boundary (exposed to outside)
Faces.on_boundary()

# Faces with specific normal direction
Faces.where(normal=(1, 0, 0))      # Facing +x direction
Faces.where(normal=(0, 0, 1), normal_tol=0.1)  # Tolerance in radians

# Faces at specific coordinate
Faces.where(x=100)                 # At x=100 plane
```

### Combined Queries

```python
# Boundary faces at x=100 (for pressure loading)
Faces.on_boundary().and_where(x=100)

# Faces on specific elements
Faces.on_elements(Elements.where(material="steel"))
```

## Set Operations

Queries support set operations via methods or operators:

### Methods

```python
q1.union(q2)        # A ∪ B - items in either query
q1.intersect(q2)    # A ∩ B - items in both queries
q1.subtract(q2)     # A - B - items in q1 but not q2
q1.invert()         # ~A - items NOT in query
```

### Operators

```python
q1 | q2    # Union (same as union)
q1 & q2    # Intersection (same as intersect)
q1 - q2    # Difference (same as subtract)
~q1        # Invert
```

### Examples

```python
# Nodes at both ends
ends = Nodes.where(x=0) | Nodes.where(x=100)

# Interior nodes (not on boundary)
interior = ~(Nodes.where(x=0) | Nodes.where(x=100))

# Nodes at x=0 but not at y=0
edge_only = Nodes.where(x=0) - Nodes.where(y=0)

# Nodes in corner (both conditions)
corner = Nodes.where(x=0) & Nodes.where(y=0)
```

## Named Components

Components allow saving and reusing query results, similar to APDL's CM (Component Manager) command.

### Defining Components

```python
model = (
    Model(mesh, materials={"steel": steel})
    .define_component("fixed_face", Nodes.where(x=0))
    .define_component("loaded_face", Nodes.where(x=100))
    .define_component("support_elements", Elements.touching(Nodes.where(z=0)))
)
```

### Using Components

```python
# Reference by name
model = model.constrain(Nodes.in_component("fixed_face"), dofs=["ux", "uy", "uz"])
model = model.load(Nodes.in_component("loaded_face"), Force(fx=1000))

# Elements from component
model = model.assign(Elements.in_component("support_elements"), material="steel")
```

### Benefits

1. **Reusability**: Define once, use multiple times
2. **Readability**: Named selections are self-documenting
3. **APDL Compatibility**: Familiar to ANSYS users

## Evaluation

Queries are evaluated against a mesh to get actual indices:

```python
# Evaluate to get numpy array of indices
fixed_indices = Nodes.where(x=0).evaluate(mesh)
print(f"Found {len(fixed_indices)} fixed nodes")

# Use in model building (evaluation happens automatically)
model = model.constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
```

## Practical Examples

### Cantilever Beam

```python
# Fixed at x=0, load at x=L
model = (
    Model(mesh, materials={"steel": steel})
    .assign(Elements.all(), material="steel")
    .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
    .load(Nodes.where(x=beam_length), Force(fz=-1000))
)
```

### Symmetry Boundary Conditions

```python
# Quarter model with symmetry
model = (
    Model(mesh, materials={"steel": steel})
    .assign(Elements.all(), material="steel")
    # Symmetry plane at x=0 (constrain ux only)
    .constrain(Nodes.where(x=0), dofs=["ux"])
    # Symmetry plane at y=0 (constrain uy only)
    .constrain(Nodes.where(y=0), dofs=["uy"])
    # Bottom support
    .constrain(Nodes.where(z=0), dofs=["uz"])
)
```

### Distributed Load

```python
# Distribute load across multiple nodes
loaded_nodes = Nodes.where(x=100).evaluate(mesh)
n_nodes = len(loaded_nodes)
force_per_node = total_force / n_nodes

model = model.load(Nodes.where(x=100), Force(fx=force_per_node))
```

### Pressure on a Face

```python
# Apply pressure to boundary face with outward normal
model = model.load(
    Faces.on_boundary().and_where(normal=(0, 0, 1)),
    Pressure(1e6)  # 1 MPa
)
```

## Query Reference

### Nodes Methods

| Method | Description |
|--------|-------------|
| `Nodes.all()` | Select all nodes |
| `Nodes.where(**kwargs)` | Select by coordinate predicates |
| `Nodes.by_indices(list)` | Select by explicit indices |
| `Nodes.near_point(point, tol)` | Nodes within distance of point |
| `Nodes.in_sphere(center, radius)` | Nodes inside sphere |
| `Nodes.in_box(min, max)` | Nodes inside axis-aligned box |
| `Nodes.near_line(start, end, tol)` | Nodes near line segment |
| `Nodes.on_elements(query)` | Nodes on selected elements |
| `Nodes.in_component(name)` | Nodes in named component |

### Elements Methods

| Method | Description |
|--------|-------------|
| `Elements.all()` | Select all elements |
| `Elements.where(**kwargs)` | Select by predicates |
| `Elements.by_indices(list)` | Select by explicit indices |
| `Elements.attached_to(nodes)` | Elements with ALL nodes in selection |
| `Elements.touching(nodes)` | Elements with ANY node in selection |
| `Elements.in_component(name)` | Elements in named component |

### Faces Methods

| Method | Description |
|--------|-------------|
| `Faces.on_boundary()` | Boundary faces only |
| `Faces.where(**kwargs)` | Select by coordinate/normal predicates |
| `Faces.on_elements(query)` | Faces on selected elements |
| `Faces.in_component(name)` | Faces in named component |

### Predicate Operators

| Suffix | Meaning | Example |
|--------|---------|---------|
| (none) | Equals | `x=0` |
| `__gt` | Greater than | `x__gt=10` |
| `__lt` | Less than | `x__lt=50` |
| `__gte` | Greater or equal | `x__gte=0` |
| `__lte` | Less or equal | `x__lte=100` |
| `__between` | In range (inclusive) | `x__between=(0, 100)` |
