# Scordelis-Lo Roof

A cylindrical barrel vault supported on rigid diaphragms at both ends, subject to self-weight. This is a classic shell benchmark for testing membrane-bending coupling in curved shell elements.

## Source

- Scordelis, A.C. and Lo, K.S., "Computer Analysis of Cylindrical Shells," ACI Journal, Vol. 61, 1964, pp. 539-561.
- MacNeal, R.H. and Harder, R.L., "A Proposed Standard Set of Problems to Test Finite Element Accuracy," Finite Elements in Analysis and Design, Vol. 1, 1985, pp. 3-20.
- Belytschko, T. et al., "Stress Projection for Membrane and Shear Locking in Shell Finite Elements," Computer Methods in Applied Mechanics and Engineering, Vol. 51, 1985, pp. 221-258.

## Problem Description

### Geometry

A portion of a cylindrical barrel vault:

```
           ^ Y (vertical)
           |
           |    /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
           |   /                          \
           |  /    Cylindrical surface     \
           | /      R = 25, θ = 40°         \
           |/________________________________\
          O----------------------------------------> X (longitudinal)
         /
        v Z (transverse)

    Cross-section (Y-Z plane):

              Y ^
                |    θ = 40° (half-angle)
                |   /
                |  /  R = 25
                | /
    ____________|/_____________

    Free edge ↓     ↓ Free edge
```

**Dimensions:**
- Radius: R = 25 (length units)
- Length: L = 50 (longitudinal span)
- Half-angle: θ = 40° (subtended angle from apex)
- Thickness: t = 0.25

**Coordinate System:**
- X: Longitudinal axis (along the length of the roof)
- Y: Vertical axis (upward)
- Z: Transverse axis

### Material Properties

Isotropic linear elastic:
- Young's modulus: E = 4.32 × 10⁸
- Poisson's ratio: ν = 0.0

Note: ν = 0 simplifies the problem by decoupling membrane and bending behavior.

### Boundary Conditions

**Rigid diaphragms at both ends (x = 0 and x = L):**
- u_x = 0 (no longitudinal displacement)
- u_z = 0 (no transverse displacement)
- θ_y = 0 (no rotation about vertical axis)
- u_y ≠ 0 (vertical displacement allowed)

**Free curved edges:**
- Both longitudinal edges (at θ = ±40°) are free

### Loading

Uniform self-weight (gravity loading):
- q = 90 (force per unit area, acting in -Y direction)

This represents a distributed load normal to the shell surface projected onto the vertical direction.

## Symmetry

The problem has two planes of symmetry:
1. **Longitudinal symmetry** at x = L/2 (transverse plane)
2. **Transverse symmetry** at z = 0 (longitudinal-vertical plane)

A quarter model can be used with appropriate symmetry boundary conditions:
- At x = L/2: u_x = 0, θ_y = 0, θ_z = 0
- At z = 0: u_z = 0, θ_x = 0, θ_y = 0

## Expected Results

**Primary target value:**

| Quantity | Location | Reference Value | Source |
|----------|----------|-----------------|--------|
| Vertical displacement u_y | Point B (midpoint of free edge: x=L/2, θ=40°) | 0.3024 | MacNeal-Harder (shear-flexible) |
| Vertical displacement u_y | Point B | 0.3006 | Shear-rigid formulation |

**Secondary verification points:**

| Quantity | Location | Description |
|----------|----------|-------------|
| u_y | Point A (apex center: x=L/2, θ=0°) | Maximum uplift at apex |
| σ_x | Point A | Longitudinal membrane stress |
| M_x | Point B | Bending moment at free edge |

## Test Parameters

### Element Types

This benchmark is designed for shell elements:

| Element Type | Notes |
|--------------|-------|
| 4-node quad (MITC4, DSG, etc.) | Standard bilinear shell |
| 8-node quad (serendipity) | Quadratic shell |
| 9-node quad (Lagrangian) | Quadratic shell with center node |
| 3-node triangle (DKT, DSG3) | Linear triangular shell |
| 6-node triangle | Quadratic triangular shell |

### Mesh Recommendations

For convergence study, use structured meshes on the quarter model:

| Mesh Level | Longitudinal × Circumferential | Elements | Notes |
|------------|--------------------------------|----------|-------|
| Coarse | 2 × 2 | 4 | Very poor accuracy expected |
| Medium | 4 × 4 | 16 | Moderate accuracy |
| Fine | 8 × 8 | 64 | Good accuracy |
| Very fine | 16 × 16 | 256 | Reference-quality |

Note: Circumferential mesh should follow the curved geometry.

### Convergence Criteria

Expected convergence behavior for well-formulated shell elements:
- Monotonic convergence to reference value
- 2% error target with medium mesh (16 elements)
- <0.5% error with fine mesh (64 elements)

## Implementation Notes

### Why This Test Matters

The Scordelis-Lo roof tests several critical shell element capabilities:

1. **Membrane-bending coupling**: The roof carries load through combined membrane action (longitudinal compression) and bending (transverse curvature). Elements must accurately couple these modes.

2. **Curved geometry**: The cylindrical surface requires accurate geometric representation. Poor curvature handling leads to "faceting" errors.

3. **Boundary condition sensitivity**: The rigid diaphragm conditions are crucial. Incorrect implementation leads to large errors.

4. **Locking resistance**: Standard displacement-based elements may exhibit membrane locking on this problem. Well-formulated elements (MITC, EAS, DSG) should not lock.

### Common Issues

1. **Membrane locking**: Low-order elements may be too stiff, underpredicting displacement
2. **Zero Poisson's ratio**: Some implementations fail when ν = 0; this is a bug
3. **Faceted geometry**: Flat-facet approximation of the cylinder adds geometric error
4. **Diaphragm conditions**: Must constrain correct DOFs; over-constraint is common

### Solid Element Alternative

While this is a shell benchmark, it can be modeled with solid elements:
- Use 2+ elements through thickness (hex8 or hex20)
- Apply diaphragm conditions on end faces
- Results should converge to shell solution as mesh is refined
- Significantly more expensive computationally

This approach is useful for validating solid element accuracy on thin structures before shell elements are available.

## Variations

### Modified Material Properties

Alternative material specification (equivalent problem):
- E = 432,000,000
- t = 0.25
- q = 90

### Unit-Consistent Version

For SI-like units:
- R = 25 m, L = 50 m, t = 0.25 m
- E = 4.32 × 10⁸ Pa
- q = 90 Pa (pressure = ρgh equivalent)
- Expected u_y = 0.3024 m

## Tolerances

- Target: ±2% of reference value (0.2963 to 0.3085)
- Acceptable for coarse mesh: ±10%
- Numerical precision: Results should be identical for symmetric mesh refinement

## References

1. Scordelis, A.C. and Lo, K.S., "Computer Analysis of Cylindrical Shells," ACI Journal, Vol. 61, 1964, pp. 539-561.
2. MacNeal, R.H. and Harder, R.L., "A Proposed Standard Set of Problems to Test Finite Element Accuracy," Finite Elements in Analysis and Design, Vol. 1, 1985, pp. 3-20.
3. Belytschko, T., Stolarski, H., Liu, W.K., Carpenter, N., and Ong, J.S., "Stress Projection for Membrane and Shear Locking in Shell Finite Elements," Computer Methods in Applied Mechanics and Engineering, Vol. 51, 1985, pp. 221-258.
4. Chapelle, D. and Bathe, K.J., "The Finite Element Analysis of Shells - Fundamentals," Springer, 2003.
