# NAFEMS Benchmark Specifications

This document provides complete specifications for implementing NAFEMS linear elastic benchmarks
for FEA verification. These benchmarks establish target values that validated FEA codes must match.

**Reference:** NAFEMS Publication TNSB, Rev. 3, "The Standard NAFEMS Benchmarks," October 1990.

---

## LE1: Plane Stress - Elliptic Membrane

### Overview

A plane stress analysis of an elliptic membrane subject to uniform tension on its outer edge.
Due to symmetry, only one quarter of the geometry is modeled.

### Geometry (2D, quarter model)

```
      y
      ^
      |     B (0, 2750)
      |    /
      |   /  outer ellipse
      |  /
  A   | /
(0,1000)
      |/__________________ x
                    C (3250, 0)
      D (2000, 0)
```

**Outer ellipse:**
- Semi-major axis (x-direction): c = 3250 mm
- Semi-minor axis (y-direction): b = 2750 mm

**Inner ellipse:**
- Semi-major axis (x-direction): d = 2000 mm
- Semi-minor axis (y-direction): a = 1000 mm

**Key points:**
| Point | Coordinates (mm) | Description |
|-------|------------------|-------------|
| A | (0, 1000) | Inner ellipse, y-axis |
| B | (0, 2750) | Outer ellipse, y-axis |
| C | (3250, 0) | Outer ellipse, x-axis |
| D | (2000, 0) | Inner ellipse, x-axis (target point) |

### Material Properties

| Property | Value | Units |
|----------|-------|-------|
| Young's modulus (E) | 210,000 | MPa |
| Poisson's ratio (ν) | 0.3 | - |
| Thickness (t) | 0.1 | m |

**Analysis type:** Plane stress (σ_z = τ_xz = τ_yz = 0)

### Boundary Conditions

| Edge | Constraint | Description |
|------|------------|-------------|
| AB (x=0, inner to outer on y-axis) | u_x = 0 | Symmetry about y-axis |
| CD (y=0, outer to inner on x-axis) | u_y = 0 | Symmetry about x-axis |

### Loading

| Location | Load Type | Value | Direction |
|----------|-----------|-------|-----------|
| Outer ellipse (arc BC) | Tension traction | 10 MPa | Normal (outward) |

### Target Solution

| Quantity | Location | Value | Units |
|----------|----------|-------|-------|
| σ_yy | Point D (2000, 0) | **92.7** | MPa |

### Notes

- The stress concentration occurs at the inner ellipse where the curvature is highest
- Higher-order elements (Quad8, Tri6) converge to values slightly below 92.7 MPa
- Mesh refinement near point D is critical for accuracy

---

## LE10: Thick Plate Under Pressure

### Overview

A 3D solid analysis of a thick elliptic plate with a central elliptic hole, subject to
uniform pressure on its upper surface. Due to symmetry, only one quarter of the geometry
is modeled.

### Geometry (3D, quarter model)

```
Top view (z = 0 mid-plane):
      y
      ^
      |     B (0, 2750, z)
      |    /
      |   /  outer ellipse
      |  /
  A   | /
(0,1000,z)
      |/__________________ x
                    C (3250, 0, z)
      D (2000, 0, z)

Side view:
      z
      ^
   +300 |-------- upper surface (pressure applied)
      0 |-------- mid-plane (z=0 reference)
   -300 |-------- lower surface
```

**Outer ellipse:**
- Semi-major axis (x-direction): c = 3250 mm
- Semi-minor axis (y-direction): b = 2750 mm

**Inner ellipse (hole):**
- Semi-major axis (x-direction): d = 2000 mm
- Semi-minor axis (y-direction): a = 1000 mm

**Plate thickness:** h = 600 mm (from z = -300 to z = +300)

**Key points (3D):**
| Point | Coordinates (mm) | Description |
|-------|------------------|-------------|
| A | (0, 1000, ±300) | Inner ellipse on y-axis |
| B | (0, 2750, ±300) | Outer ellipse on y-axis |
| C | (3250, 0, ±300) | Outer ellipse on x-axis |
| D | (2000, 0, 300) | Inner ellipse on x-axis, **top surface** (target) |
| A' | (0, 1000, -300) | A on bottom surface |
| B' | (0, 2750, -300) | B on bottom surface |
| C' | (3250, 0, -300) | C on bottom surface |
| D' | (2000, 0, -300) | D on bottom surface |

### Material Properties

| Property | Value | Units |
|----------|-------|-------|
| Young's modulus (E) | 210,000 | MPa |
| Poisson's ratio (ν) | 0.3 | - |

**Analysis type:** 3D linear elastic solid

### Boundary Conditions

| Face/Surface | Constraint | Description |
|--------------|------------|-------------|
| Face DCD'C' (y = 0 plane) | u_y = 0 | Symmetry about x-z plane |
| Face ABA'B' (x = 0 plane) | u_x = 0 | Symmetry about y-z plane |
| Face BCB'C' (outer curved) | u_x = 0, u_y = 0 | Outer edge restrained |
| Mid-plane (z = 0) | u_z = 0 | Anti-symmetry in z |

**Note:** The boundary condition on face BCB'C' (u_x = u_y = 0) is unusual but specified
in the NAFEMS benchmark to create a specific stress state.

### Loading

| Location | Load Type | Value | Direction |
|----------|-----------|-------|-----------|
| Upper surface (z = +300) | Pressure | 1 MPa | Downward (-z) |

### Target Solution

| Quantity | Location | Value | Units |
|----------|----------|-------|-------|
| σ_yy | Point D (2000, 0, 300) | **-5.38** | MPa |

**Note:** The negative sign indicates compressive stress in the y-direction.

### Notes

- Point D is on the **upper surface** (z = +300) at the inner ellipse edge
- The target stress was corrected in NAFEMS TNSB Revision 3
- Mesh refinement near point D and through the thickness is important
- Second-order elements (Hex20, Tet10, Wedge15) recommended for accuracy

---

## Implementation Checklist

### LE1 Implementation (mops-43g.6)

- [ ] Create parametric mesh generator for elliptic membrane
- [ ] Support multiple element types (Tri3, Tri6, Quad4, Quad8)
- [ ] Apply plane stress material (2D constitutive matrix)
- [ ] Apply boundary conditions (symmetry on AB, CD)
- [ ] Apply edge traction on outer ellipse
- [ ] Extract σ_yy at point D
- [ ] Compare to target: 92.7 MPa (tolerance: ±2%)
- [ ] Document mesh convergence study

### LE10 Implementation (mops-43g.7)

- [ ] Create parametric mesh generator for thick elliptic plate
- [ ] Support multiple element types (Tet4, Tet10, Hex8, Hex20)
- [ ] Apply 3D isotropic material
- [ ] Apply all boundary conditions:
  - Symmetry on x=0 and y=0 planes
  - Outer edge constraint (u_x = u_y = 0)
  - Mid-plane constraint (u_z = 0)
- [ ] Apply pressure on upper surface
- [ ] Extract σ_yy at point D (2000, 0, 300)
- [ ] Compare to target: -5.38 MPa (tolerance: ±2%)
- [ ] Document mesh convergence study

---

## Mesh Generation Guidance

### LE1 (2D)

For parametric mesh generation using Gmsh:

```python
# Ellipse parameters
a, b = 1000, 2750  # Inner/outer y semi-axes (mm)
c, d = 3250, 2000  # Outer/inner x semi-axes (mm)

# Key points
A = (0, a)      # Inner on y-axis
B = (0, b)      # Outer on y-axis
C = (c, 0)      # Outer on x-axis
D = (d, 0)      # Inner on x-axis (TARGET)

# Mesh density near D for stress accuracy
mesh_size_D = 50  # Fine mesh at target point
mesh_size_far = 200  # Coarser away from D
```

### LE10 (3D)

For parametric mesh generation using Gmsh:

```python
# Same ellipse parameters as LE1
a, b = 1000, 2750
c, d = 3250, 2000
h = 600  # Plate thickness (mm)

# Key point for stress extraction
D = (d, 0, h/2)  # (2000, 0, 300)

# Through-thickness divisions (minimum 4 for Hex8)
n_layers = 4  # For Hex8
# or n_layers = 2 for Hex20 (quadratic captures bending)
```

---

## Acceptance Criteria

Both benchmarks should achieve results within **±2%** of target values using:

| Benchmark | Element Types | Target | Tolerance |
|-----------|---------------|--------|-----------|
| LE1 | Quad8, Tri6 (2nd order) | 92.7 MPa | 91.0 - 94.4 MPa |
| LE10 | Hex20, Tet10 (2nd order) | -5.38 MPa | -5.49 to -5.27 MPa |

First-order elements (Tri3, Quad4, Tet4, Hex8) may require finer meshes to achieve
comparable accuracy due to their limited strain representation.

---

## References

1. NAFEMS, "The Standard NAFEMS Benchmarks," TNSB Rev. 3, October 1990
2. NAFEMS, "Background to Benchmarks," R0006, 1993
3. Abaqus Benchmarks Manual, Section 4.2.1 (LE1) and 4.2.10 (LE10)
4. seamplex/feenox: Open-source implementation with Gmsh geometry files
