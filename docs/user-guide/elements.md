# Element Library

MOPS provides a comprehensive library of finite elements for 3D solid, 2D plane, and axisymmetric analysis. All elements use standard isoparametric formulations with Gauss quadrature integration.

## Overview

| Element | Type | Nodes | DOFs | Integration | Primary Use |
|---------|------|-------|------|-------------|-------------|
| `tet4` | 3D solid | 4 | 12 | 1-point | General meshing |
| `tet10` | 3D solid | 10 | 30 | 4-point | Higher accuracy |
| `hex8` | 3D solid | 8 | 24 | 2×2×2 | Structured meshes |
| `hex8sri` | 3D solid | 8 | 24 | 1 + 2×2×2 | Bending-dominated |
| `hex8bbar` | 3D solid | 8 | 24 | 2×2×2 | Near-incompressible |
| `hex20` | 3D solid | 20 | 60 | 3×3×3 | High accuracy structured |
| `tri3` | 2D | 3 | 6 | 1-point | Plane stress/strain |
| `quad4` | 2D | 4 | 8 | 2×2 | Plane stress/strain |

## 3D Solid Elements

### Tet4 (Linear Tetrahedron)

The simplest 3D element, suitable for automatic mesh generation of complex geometries.

```
        3
       /|\
      / | \
     /  |  \
    /   0   \
   /  .' '.  \
  1---------2
```

**Characteristics:**
- 4 corner nodes, 3 DOFs per node (ux, uy, uz)
- Constant strain throughout element
- Single integration point at centroid
- Best for: Coarse preliminary analysis, complex geometry

**Limitations:**
- Requires fine meshes for accuracy
- Volumetric locking with nearly incompressible materials (ν → 0.5)
- Shear locking in bending-dominated problems

**Node numbering:** Counter-clockwise when viewed from opposite vertex.

```python
mesh = Mesh(nodes, elements, "tet4")
```

### Tet10 (Quadratic Tetrahedron)

Higher-order tetrahedron with curved edges. Recommended for production analysis.

```
        3
       /|\
      7 | 8
     /  9  \
    /   0   \
   4   ' '   6
  / '       ' \
 1------5------2
```

**Characteristics:**
- 10 nodes: 4 vertices + 6 edge midpoints
- Linear strain variation (quadratic displacement)
- 4-point Gauss quadrature
- Best for: Complex geometry with accuracy requirements

**Advantages over Tet4:**
- Much better accuracy per element
- Better representation of curved surfaces
- Handles bending better

```python
mesh = Mesh(nodes, elements, "tet10")
```

### Hex8 (Linear Hexahedron)

Brick element for structured meshes. More accurate than Tet4 for the same mesh density.

```
      7-------6
     /|      /|
    / |     / |
   4-------5  |
   |  3----|--2
   | /     | /
   |/      |/
   0-------1
```

**Characteristics:**
- 8 corner nodes, 3 DOFs per node
- Trilinear interpolation
- 2×2×2 Gauss quadrature (8 points)
- Best for: Rectangular geometries, layered structures

**Node numbering:** Bottom face CCW (0-1-2-3), then top face CCW (4-5-6-7).

```python
mesh = Mesh(nodes, elements, "hex8")
```

**Limitations:**
- Shear locking in bending-dominated problems (use `hex8sri` instead)
- Volumetric locking with nearly incompressible materials (use `hex8bbar` instead)

### Hex8SRI (Selective Reduced Integration)

A variant of Hex8 that mitigates **shear locking** through selective reduced integration. Use this for bending-dominated problems or coarse meshes.

**What is Shear Locking?**

Shear locking occurs when linear elements cannot represent pure bending without introducing spurious shear strains. This makes the element artificially stiff, underestimating displacements and causing inaccurate stresses.

```
Pure bending should have:     But Hex8 predicts:
- Quadratic displacement      - Linear displacement (locked)
- Zero shear strain           - Non-zero spurious shear
```

**How Hex8SRI Fixes It:**

Hex8SRI uses different integration orders for volumetric and deviatoric (shear) strain:
- **1-point** integration at element center for volumetric strain
- **2×2×2** integration for deviatoric strain

This eliminates spurious shear stiffness while maintaining element stability.

**Characteristics:**
- Same 8 nodes and DOFs as Hex8
- No hourglass modes (unlike uniform reduced integration)
- Significantly improved bending accuracy
- Similar computational cost to standard Hex8

**When to use Hex8SRI:**
- Bending-dominated problems (cantilevers, plates, beams)
- Coarse meshes where shear locking would be significant
- When mesh refinement is constrained by computational resources
- Solid-meshed shell-like structures

```python
mesh = Mesh(nodes, elements, "hex8sri")
```

**Example: Cantilever Beam**

```python
# Hex8SRI is more accurate than Hex8 for bending
# On a coarse mesh (5x1x1 elements):
#   - Hex8 error: ~40-50% (too stiff)
#   - Hex8SRI error: ~20-30% (much better)
#   - Both converge with mesh refinement
```

### Hex8Bbar (Mean Dilatation / B-bar)

A variant of Hex8 that addresses **volumetric locking** through the B-bar (mean dilatation) formulation. Use this for nearly incompressible materials (ν → 0.5).

**What is Volumetric Locking?**

Volumetric locking occurs when elements overconstrain volumetric deformation for materials with high Poisson's ratio (rubber, biological tissue, incompressible fluids). The element becomes artificially stiff, severely underestimating displacements.

**How Hex8Bbar Fixes It:**

Hex8Bbar replaces local volumetric strain with an element-averaged volumetric strain:
```
Standard: ε_vol at each point
B-bar:    ε_vol_avg = (1/V) ∫ ε_vol dV (averaged over element)
```

This prevents over-constraining volumetric deformation.

**Characteristics:**
- Same 8 nodes and DOFs as Hex8
- Addresses both volumetric and shear locking
- Mathematically rigorous (variational basis)
- Widely used in metal plasticity

**When to use Hex8Bbar:**
- Nearly incompressible materials (rubber, ν ≈ 0.499)
- Biological tissue modeling
- Metal forming simulations (large plastic deformation)
- Problems where both volumetric and shear locking are concerns

```python
mesh = Mesh(nodes, elements, "hex8bbar")
```

### Choosing Between Hex8 Variants

| Situation | Element | Reason |
|-----------|---------|--------|
| General structured mesh | `hex8` | Safe default, well-understood |
| Bending-dominated (plates, beams) | `hex8sri` | Reduces shear locking |
| Coarse mesh, bending loads | `hex8sri` | Better accuracy with fewer elements |
| Rubber, ν ≈ 0.5 | `hex8bbar` | Prevents volumetric locking |
| Metal forming | `hex8bbar` | Standard for plasticity |
| High accuracy, any problem | `hex20` | Quadratic elements avoid locking |

**Decision Flowchart:**

```
Is the material nearly incompressible (ν > 0.45)?
  └─ Yes → Use hex8bbar
  └─ No → Is the problem bending-dominated?
           └─ Yes → Use hex8sri (coarse mesh) or hex8 (fine mesh)
           └─ No → Use hex8 (default)
```

**Note:** When in doubt, `hex20` (quadratic elements) avoids both locking issues but costs more computationally.

### Hex20 (Quadratic Hexahedron)

High-order hexahedron for maximum accuracy in structured meshes.

```
      7---14---6
     /|       /|
   15 |     13 |
   /  19   /   18
  4---12---5   |
  |   3---10---2
  16 /    17  /
  | 11    | 9
  |/      |/
  0---8---1
```

**Characteristics:**
- 20 nodes: 8 vertices + 12 edge midpoints
- Serendipity shape functions (quadratic)
- 3×3×3 Gauss quadrature (27 points)
- Best for: High-accuracy structured analysis

**Advantages:**
- Excellent accuracy for bending problems
- Better curved surface representation
- Worth the computational cost for precision work

```python
mesh = Mesh(nodes, elements, "hex20")
```

## 2D Elements

2D elements are used for plane stress, plane strain, and axisymmetric analyses. They have 2 DOFs per node (ux, uy).

### Plane Stress

For thin structures where stress perpendicular to the plane is zero (σ_z = 0).

**Use cases:**
- Thin plates under in-plane loading
- Membrane structures
- Sheet metal analysis

```python
# Currently exposed through mops-core
# Python API integration planned
```

### Plane Strain

For long prismatic structures where strain perpendicular to the plane is zero (ε_z = 0).

**Use cases:**
- Dam cross-sections
- Tunnel walls
- Long pipe sections
- Retaining walls

```python
# Currently exposed through mops-core
# Python API integration planned
```

### Axisymmetric

For bodies of revolution loaded symmetrically about the axis. Models a 2D cross-section that represents a 3D solid.

**Use cases:**
- Pressure vessels
- Circular plates
- Shafts and pipes
- Flanges

```python
# Currently exposed through mops-core
# Python API integration planned
```

## Choosing the Right Element

### For General 3D Problems

| Situation | Recommended Element | Mesh Density |
|-----------|---------------------|--------------|
| Initial design/coarse analysis | `hex8` or `tet4` | 3-5 elements through thickness |
| Bending-dominated, coarse mesh | `hex8sri` | Fewer elements needed |
| Near-incompressible materials | `hex8bbar` | Required for ν > 0.45 |
| Production analysis | `tet10` | Auto-meshing tools |
| High accuracy requirements | `hex20` or `tet10` | Refined at stress concentrations |
| Complex geometry (auto-mesh) | `tet10` | Mesh generators like Gmsh |
| Structured/regular geometry | `hex8` or `hex20` | Manual mesh generation |

### Mesh Convergence

Always perform mesh convergence studies for critical results:

```python
# Example: Converging tip displacement
mesh_sizes = [10, 20, 40, 80]  # elements along length
for n in mesh_sizes:
    mesh = create_mesh(nx=n)
    model = build_model(mesh)
    results = solve(model)
    print(f"Elements: {mesh.n_elements}, Max disp: {results.max_displacement():.6e}")
```

Target: Results should stabilize (< 5% change) with mesh refinement.

## Element Quality

Poor element quality degrades accuracy. Check for:

1. **Aspect ratio**: Keep below 10:1 (ideally < 3:1)
2. **Jacobian ratio**: All positive, ideally > 0.3
3. **Skewness**: Angles close to ideal (60° for tri, 90° for quad)

For hexahedra, avoid highly distorted elements. For tetrahedra, avoid sliver elements (very flat or thin).

## Integration Points

Elements use Gauss quadrature for numerical integration:

| Element | Points | Location | Weights |
|---------|--------|----------|---------|
| `tet4` | 1 | Centroid | 1/6 |
| `tet10` | 4 | Interior | 1/24 each |
| `hex8` | 8 | ±1/√3 in each direction | 1.0 each |
| `hex8sri` | 1 + 8 | Center (vol) + ±1/√3 (dev) | special |
| `hex8bbar` | 8 | ±1/√3 in each direction | 1.0 each |
| `hex20` | 27 | ±√(3/5), 0 in each direction | varies |

Stresses are most accurate at integration points. Element-averaged stresses (reported by MOPS) are computed from integration point values.

## Material Assignment

All elements require material properties:

```python
steel = Material("steel", e=200e9, nu=0.3)
aluminum = Material("aluminum", e=70e9, nu=0.33)

model = (
    Model(mesh, materials={"steel": steel, "aluminum": aluminum})
    .assign(Elements.all(), material="steel")
    # Or assign different materials to different regions
    .assign(Elements.touching(Nodes.where(x__lt=50)), material="aluminum")
    .assign(Elements.touching(Nodes.where(x__gte=50)), material="steel")
)
```

## Stress Output

For 3D elements, stress is output in Voigt notation:

```python
stress = results.stress()  # Shape: (n_elements, 6)
# Components: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]

# Access individual components
sigma_xx = stress[:, 0]
sigma_yy = stress[:, 1]
sigma_zz = stress[:, 2]
tau_xy = stress[:, 3]
tau_yz = stress[:, 4]
tau_xz = stress[:, 5]
```

Derived quantities:

```python
# Von Mises (equivalent) stress - for ductile failure
von_mises = results.von_mises()

# Principal stresses (eigenvalues of stress tensor)
# Available via results.principal_stresses() if implemented
```

## Verification Examples

### Uniaxial Tension

Simple test to verify element implementation:

```python
# Bar under tension: σ = F/A
# Expected: Uniform stress, linear displacement
results = solve(bar_model)
stress = results.stress()
print(f"Expected: 100 MPa, Got: {np.mean(stress[:, 0]):.1f} MPa")
```

### Patch Tests

Elements pass patch tests:
1. **Rigid body motion**: Zero stress under translation/rotation
2. **Constant strain**: Exact recovery of uniform strain field

### Cantilever Beam

Classical benchmark with analytical solution:

```python
# Tip deflection: δ = P*L³/(3*E*I)
delta_analytical = P * L**3 / (3 * E * I)
delta_fea = results.max_displacement()
error = abs(delta_fea - delta_analytical) / delta_analytical
print(f"Error: {error*100:.2f}%")
```

Expected: < 5% error with adequate mesh refinement.

## Next Steps

- [Getting Started](getting-started.md): Run your first analysis
- [Results](results.md): Working with analysis results
- [Visualization](visualization.md): Plotting stress fields
