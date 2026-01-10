# Element Library Design

**Date:** 2026-01-10
**Status:** Draft
**Depends on:** [MOPS Initial Design](2026-01-10-mops-initial-design.md)

## Overview

This document specifies the finite element formulations for MOPS. All elements implement the `Element` trait defined in `mops-core/src/element.rs`.

## Element Catalog

### Initial Target (Linear Static)

| Element | Type | Nodes | DOFs | Integration | Use Case |
|---------|------|-------|------|-------------|----------|
| Tet4 | 3D solid | 4 | 12 | 1-point | General 3D, auto-meshing |
| Tet10 | 3D solid | 10 | 30 | 4-point | Higher accuracy 3D |
| Hex8 | 3D solid | 8 | 24 | 2×2×2 | Structured meshes |
| Hex20 | 3D solid | 20 | 60 | 3×3×3 | Higher accuracy structured |
| PlaneStress | 2D | 3 or 4 | 6 or 8 | varies | Thin plates (σ_z = 0) |
| PlaneStrain | 2D | 3 or 4 | 6 or 8 | varies | Long prismatic (ε_z = 0) |
| Axisymmetric | 2D | 3 or 4 | 6 or 8 | varies | Bodies of revolution |

### Priority Order

1. **Tet4** - Simplest 3D element, validates assembly pipeline
2. **Hex8** - Higher accuracy reference for verification
3. **Tet10** - Production workhorse for complex geometry
4. **PlaneStress/PlaneStrain** - 2D verification problems
5. **Axisymmetric** - Specialized applications
6. **Hex20** - High-order structured meshes

## 3D Solid Elements

### Common Formulation

For 3D solid elements with 3 DOFs per node (u, v, w displacements):

**Strain-displacement relation (small strain):**
```
ε = [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz]^T
ε = B * u
```

**Constitutive relation (isotropic linear elastic):**
```
σ = D * ε
```
where D is the 6×6 constitutive matrix from `material.constitutive_3d()`.

**Element stiffness:**
```
K_e = ∫∫∫ B^T * D * B dV
```

For isoparametric elements, integration is performed in natural coordinates (ξ, η, ζ):
```
K_e = ∫∫∫ B^T * D * B * |J| dξ dη dζ
```

### Tet4 (4-Node Tetrahedron)

**Shape functions (constant strain):**
```
N_i = (a_i + b_i*x + c_i*y + d_i*z) / (6V)
```
where V is the element volume and coefficients derive from nodal coordinates.

**Properties:**
- Constant strain/stress within element
- Single integration point at centroid
- Volume: V = |det([x2-x1, x3-x1, x4-x1])| / 6
- Exact integration (no numerical quadrature needed)

**B-matrix (6×12):**
```
B = [b1  0  0  b2  0  0  b3  0  0  b4  0  0 ]
    [ 0 c1  0   0 c2  0   0 c3  0   0 c4  0 ]
    [ 0  0 d1   0  0 d2   0  0 d3   0  0 d4]
    [c1 b1  0  c2 b2  0  c3 b3  0  c4 b4  0 ]
    [ 0 d1 c1   0 d2 c2   0 d3 c3   0 d4 c4]
    [d1  0 b1  d2  0 b2  d3  0 b3  d4  0 b4]
    / (6V)
```

**Stiffness matrix:**
```
K = V * B^T * D * B
```

**Known limitations:**
- Volumetric locking in nearly incompressible materials (ν → 0.5)
- Requires fine meshes for accuracy
- Shear locking in bending-dominated problems

### Tet10 (10-Node Tetrahedron)

**Shape functions (quadratic):**
Quadratic interpolation with nodes at vertices (1-4) and edge midpoints (5-10).

**Properties:**
- Linear strain variation within element
- 4-point Gauss quadrature (exact for quadratic)
- Much better accuracy than Tet4 for same mesh density

**Integration points (barycentric coordinates):**
```
α = (5 + 3√5) / 20 ≈ 0.5854
β = (5 - √5) / 20 ≈ 0.1382
Weight = 1/24 each

Point 1: (α, β, β, β)
Point 2: (β, α, β, β)
Point 3: (β, β, α, β)
Point 4: (β, β, β, α)
```

### Hex8 (8-Node Hexahedron)

**Shape functions (trilinear):**
```
N_i = (1 + ξ_i*ξ)(1 + η_i*η)(1 + ζ_i*ζ) / 8
```
where (ξ_i, η_i, ζ_i) are ±1 for node i.

**Properties:**
- Linear displacement interpolation in each direction
- 2×2×2 Gauss quadrature (8 points)
- Better accuracy than Tet4 for structured meshes
- Can suffer from shear locking in bending

**Integration points:**
```
ξ, η, ζ ∈ {-1/√3, +1/√3}
Weight = 1.0 for each point
```

**Jacobian:**
```
J = [∂x/∂ξ  ∂y/∂ξ  ∂z/∂ξ]
    [∂x/∂η  ∂y/∂η  ∂z/∂η]
    [∂x/∂ζ  ∂y/∂ζ  ∂z/∂ζ]
```

### Hex20 (20-Node Hexahedron)

**Shape functions (serendipity quadratic):**
20 nodes: 8 vertices + 12 edge midpoints.

**Properties:**
- Quadratic displacement interpolation
- 3×3×3 Gauss quadrature (27 points)
- Higher accuracy for bending and stress concentration

## 2D Elements

### Common Formulation

2D elements use either plane stress or plane strain assumptions.

**Plane stress (thin plates, σ_z = 0):**
- Use 3×3 constitutive matrix from `material.constitutive_plane_stress()`
- Thickness t is a parameter

**Plane strain (long prismatic, ε_z = 0):**
- Use 3×3 constitutive matrix from `material.constitutive_plane_strain()`
- Unit thickness assumed

**Strain vector:**
```
ε = [ε_xx, ε_yy, γ_xy]^T
```

### Tri3 (3-Node Triangle)

**Shape functions (area coordinates):**
```
N_i = A_i / A_total
```

**Properties:**
- Constant strain (CST element)
- Single integration point
- 2 DOFs per node (u, v)

### Quad4 (4-Node Quadrilateral)

**Shape functions (bilinear):**
```
N_i = (1 + ξ_i*ξ)(1 + η_i*η) / 4
```

**Properties:**
- 2×2 Gauss quadrature
- Better accuracy than Tri3

### Axisymmetric Elements

For bodies of revolution about the z-axis.

**Strain vector (4 components):**
```
ε = [ε_rr, ε_zz, ε_θθ, γ_rz]^T
```

**Hoop strain:**
```
ε_θθ = u_r / r
```

**Integration:**
Volume element is 2πr dr dz, so integration includes r factor.

## Implementation Structure

### Module Layout

```
mops-core/src/
├── element.rs              # Element trait (existing)
├── element/
│   ├── mod.rs              # Re-exports, ElementType → Element dispatch
│   ├── gauss.rs            # Gauss quadrature tables
│   ├── tet4.rs             # Tet4 implementation
│   ├── tet10.rs            # Tet10 implementation
│   ├── hex8.rs             # Hex8 implementation
│   ├── hex20.rs            # Hex20 implementation
│   ├── plane_stress.rs     # Tri3/Quad4 plane stress
│   ├── plane_strain.rs     # Tri3/Quad4 plane strain
│   └── axisymmetric.rs     # Axisymmetric elements
```

### Element Dispatch

```rust
/// Create element implementation from mesh element type.
pub fn create_element(element_type: ElementType) -> Box<dyn Element> {
    match element_type {
        ElementType::Tet4 => Box::new(Tet4),
        ElementType::Tet10 => Box::new(Tet10),
        ElementType::Hex8 => Box::new(Hex8),
        ElementType::Hex20 => Box::new(Hex20),
        // 2D elements need additional context (thickness, etc.)
        _ => unimplemented!(),
    }
}
```

### Gauss Quadrature Module

```rust
/// Gauss quadrature point and weight.
pub struct GaussPoint {
    pub coords: [f64; 3],  // Natural coordinates (ξ, η, ζ)
    pub weight: f64,
}

/// Standard quadrature rules.
pub fn gauss_1d(n: usize) -> Vec<(f64, f64)>;  // (point, weight) pairs
pub fn gauss_tet(n: usize) -> Vec<GaussPoint>; // Tetrahedral rules
pub fn gauss_hex(n: usize) -> Vec<GaussPoint>; // Hexahedral rules (n per direction)
```

## Verification Strategy

### Patch Tests

Every element must pass patch tests verifying:
1. **Rigid body motion** - Zero strain for pure translation/rotation
2. **Constant strain** - Exact strain for linear displacement field

### Analytical Benchmarks

| Benchmark | Elements | Reference |
|-----------|----------|-----------|
| Cantilever beam | Hex8, Tet4 | Beam theory (δ = PL³/3EI) |
| Plate with hole | Tet10, Hex8 | Kirsch solution |
| Pressurized cylinder | Axisym | Lamé equations |
| Simply supported plate | Quad4 | Plate theory |

### NAFEMS Benchmarks

| ID | Problem | Elements | Reference Value |
|----|---------|----------|-----------------|
| LE1 | Elliptic membrane | 2D | σ_y = 92.7 MPa at point D |
| LE10 | Thick plate | 3D | σ_y = 5.38 MPa at point D |

### Element Stiffness Verification

Compare computed stiffness matrices against:
1. Published element matrices (textbooks)
2. Reference FEA software (ANSYS, Abaqus)
3. Symbolic computation (for simple elements)

## Testing Infrastructure

### Unit Tests (per element)

```rust
#[test]
fn tet4_volume_unit_tetrahedron() {
    // Unit tetrahedron: V = 1/6
}

#[test]
fn tet4_stiffness_symmetric() {
    // K = K^T
}

#[test]
fn tet4_patch_test_constant_strain() {
    // Linear displacement → constant strain → exact stress
}
```

### Integration Tests

Full solve pipeline:
1. Create mesh with known geometry
2. Apply boundary conditions
3. Solve system
4. Compare displacement/stress to analytical solution

### Benchmark Tests

Longer-running NAFEMS verification suite in `tests/verification/`.

## Open Questions

1. **Reduced integration:** Add selectively reduced integration for locking relief?
2. **B-bar formulation:** Implement for volumetric locking in Tet4?
3. **Element quality metrics:** Jacobian ratio, aspect ratio warnings?

## References

- Hughes, T.J.R. "The Finite Element Method" (2000)
- Zienkiewicz & Taylor, "The Finite Element Method" Vol 1-3 (2013)
- NAFEMS Benchmark Challenge Tests
