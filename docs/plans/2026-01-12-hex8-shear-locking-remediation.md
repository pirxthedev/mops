# Hex8 Shear Locking Remediation

**Date:** 2026-01-12
**Status:** Phase 1 & 2 Complete (Hex8SRI, Hex8Bbar implemented)
**Depends on:** [Element Library Design](2026-01-10-element-library-design.md)
**Related Issue:** mops-cb0

## Overview

This document specifies strategies for mitigating shear locking in Hex8 elements. Shear locking causes the element to be overly stiff in bending-dominated problems, resulting in underestimated displacements and inaccurate stresses.

## Problem Statement

### What is Shear Locking?

Shear locking occurs when linear (trilinear for 3D) elements cannot represent pure bending deformation without introducing spurious shear strains. In pure bending:
- The actual shear strain γ_xz should be zero through thickness
- Trilinear shape functions cannot represent this state exactly
- The element "locks" by predicting non-zero shear strain, resisting deformation

### Mathematical Root Cause

For a Hex8 element in bending:
```
True bending: u(z) = κz, w(x) = -κx²/2
True shear strain: γ_xz = ∂u/∂z + ∂w/∂x = κ - κx = 0 (at neutral axis)

Hex8 approximation: u ≈ trilinear, w ≈ trilinear
Interpolated: γ_xz ≠ 0 (cannot represent quadratic w)
```

The full 2×2×2 integration captures this spurious shear strain, creating artificial stiffness.

### Observed Symptoms

From NAFEMS LE10 benchmark testing:
- Coarse hex8 mesh: -4.3 MPa vs target -5.38 MPa (20% under-displacement, 20% stress error)
- Fine hex8 mesh: -5.0 MPa (8% error, improved but not eliminated)
- Hex20 mesh: Converges with fewer elements (no shear locking)

From cantilever beam testing:
- Hex8 reaches ~90% of analytical tip deflection (10% too stiff)
- Requires mesh refinement to approach analytical solution

## Remediation Techniques

### 1. Selective Reduced Integration (Recommended - Priority 1)

**Concept:** Use different integration orders for volumetric and deviatoric strain components.

**Implementation:**
- Volumetric strain (tr(ε)/3): 1-point integration at element center
- Deviatoric strain (shear): Full 2×2×2 integration

**Advantages:**
- No hourglass modes (unlike uniform reduced integration)
- Preserves element stability
- Modest implementation complexity
- Well-established in commercial codes (ABAQUS C3D8R with hourglass control)

**Code Changes Required:**
1. Split B-matrix into volumetric and deviatoric parts: B = B_vol + B_dev
2. Integrate B_vol at single center point
3. Integrate B_dev at 2×2×2 points
4. Combine for total stiffness: K = K_vol + K_dev

**Mathematical Formulation:**
```
B_vol = (1/3) * m * m^T * B  where m = [1, 1, 1, 0, 0, 0]^T
B_dev = B - B_vol

K = ∫ B_vol^T D B_vol |J| dV (1-point) + ∫ B_dev^T D B_dev |J| dV (8-point)
```

### 2. B-bar (Mean Dilatation) Formulation (Priority 2)

**Concept:** Replace volumetric strain with element-averaged volumetric strain.

**Implementation:**
- Compute average volumetric strain: ε_vol_avg = (1/V) ∫ tr(ε)/3 dV
- Replace local ε_vol with ε_vol_avg in constitutive relation

**Advantages:**
- Addresses both shear locking and volumetric locking (incompressible materials)
- Mathematically rigorous
- Used extensively in metal plasticity

**Code Changes Required:**
1. Compute element volume and average dilatation
2. Modify B-matrix to use averaged volumetric terms
3. Assemble modified stiffness matrix

**Mathematical Formulation:**
```
B̄ = B_dev + B̄_vol

B̄_vol = (1/V) ∫ B_vol dV  (element-averaged)
```

### 3. Assumed Natural Strain (ANS) - Priority 3

**Concept:** Sample shear strains at optimal locations (element faces/edges) and interpolate.

**Implementation (for transverse shear):**
- Sample γ_xz at midpoints of faces perpendicular to z-axis
- Sample γ_yz at midpoints of faces perpendicular to y-axis
- Use bilinear interpolation within element

**Advantages:**
- Directly targets shear locking mechanism
- Preserves element stability
- Effective for shell and plate-like bending

**Complexity:** Higher than SRI or B-bar due to specialized sampling.

### 4. Enhanced Assumed Strain (EAS) - Priority 4

**Concept:** Add incompatible strain modes to enrich element behavior.

**Implementation:**
- Augment strain field: ε = ε_compatible + ε_enhanced
- Enhanced modes orthogonal to stress at element level
- Additional DOFs (internal, condensed out)

**Advantages:**
- Most flexible approach
- Can eliminate both shear and volumetric locking
- Mathematically elegant (variational basis)

**Complexity:** Highest - requires managing internal DOFs and static condensation.

### 5. Incompatible Modes (Wilson Element)

**Concept:** Add extra displacement modes that are discontinuous across element boundaries.

**Implementation:**
- Add 9 internal DOF modes (3 per direction)
- Static condensation removes internal DOFs

**Note:** Superseded by EAS in modern formulations. Not recommended for new implementation.

## Recommended Implementation Order

1. **Selective Reduced Integration** - Best effort/benefit ratio
   - Start with SRI as default for Hex8
   - Maintain full integration as option for comparison

2. **B-bar Formulation** - For volumetric + shear locking
   - Useful when ν → 0.5 (rubber, incompressible flow)
   - Complements SRI

3. **ANS** - If SRI insufficient for specific problems
   - More targeted at thin-walled structures

4. **EAS** - Research/advanced option
   - Complete locking elimination
   - Higher implementation cost

## API Design

### Element Variants

Option A: Separate element types (preferred for clarity)
```rust
pub struct Hex8;           // Standard 2×2×2 integration
pub struct Hex8SRI;        // Selective reduced integration
pub struct Hex8Bbar;       // B-bar formulation

// In Python
model.element("Hex8")      # Standard
model.element("Hex8SRI")   # Selective reduced
```

Option B: Single element with integration scheme parameter
```rust
pub enum Hex8Integration {
    Full,           // 2×2×2 for all
    Selective,      // SRI
    Bbar,           // Mean dilatation
}

impl Hex8 {
    pub fn with_integration(scheme: Hex8Integration) -> Self;
}
```

### Recommendation

Use Option A for initial implementation - cleaner API, easier testing, clear semantics. Can consolidate later if needed.

## Verification Strategy

### Locking-Sensitive Benchmarks

1. **Cantilever beam in bending**
   - Single element through thickness (worst case)
   - Compare tip deflection vs. beam theory
   - Expected: SRI should match analytical within 5%

2. **Cook's membrane**
   - Trapezoidal cantilever under shear
   - Known reference: δ_y = 23.96 at top-right corner
   - Tests combined bending and shear

3. **Scordelis-Lo roof**
   - Cylindrical shell under self-weight
   - Sensitive to membrane locking
   - Reference: δ_z = 0.3024 at midpoint

4. **NAFEMS LE10 (existing)**
   - Thick plate bending
   - Should see improved stress accuracy with SRI

### Patch Tests

SRI/B-bar elements MUST pass standard patch tests:
1. Rigid body motion (zero strain energy)
2. Constant strain state (exact reproduction)

### Hourglass Mode Check

For reduced integration variants:
1. Apply pure hourglass mode deformation
2. Verify non-zero strain energy (no rank deficiency)

## Implementation Checklist

### Phase 1: Selective Reduced Integration

- [x] Implement `Hex8SRI` struct
- [x] Add B-matrix decomposition (B_vol, B_dev)
- [x] 1-point volumetric integration
- [x] 2×2×2 deviatoric integration
- [x] Unit tests (patch tests, symmetry)
- [x] Add to Python bindings
- [x] Cook's membrane verification test (tests/verification/test_cooks_membrane.py)
- [x] Cantilever beam verification test (tests/verification/test_cantilever.py::TestCantileverHex8SRI)
- [x] NAFEMS LE10 comparison test (tests/verification/test_nafems_le10.py::TestNAFEMSLE10Hex8SRI)

### Phase 2: B-bar Formulation

- [x] Implement `Hex8Bbar` struct
- [x] Average dilatation computation
- [x] Modified B-matrix assembly
- [x] Unit tests
- [x] Nearly-incompressible test case (ν=0.499)
- [x] Add to Python bindings

### Phase 3: Documentation

- [ ] Update element library design doc
- [ ] User guide: when to use each variant
- [ ] Benchmark results documentation

## Performance Considerations

### SRI Performance

- Slightly faster than full integration (1+8 vs 8 point evaluations)
- But requires B-matrix split at each point
- Net: approximately same cost as full integration

### B-bar Performance

- Additional averaging step (small overhead)
- O(N) in number of integration points
- Negligible compared to solve time

### Memory

- No additional storage required for SRI or B-bar
- Same element connectivity and DOF count

## Detailed Technical Formulations

### Static Condensation for Enhanced/Incompatible Modes

For EAS or incompatible modes methods, internal DOFs (α) are condensed at element level:

```
System with enhanced modes:
[K_uu  K_uα] [u]   [f]
[K_αu  K_αα] [α] = [0]

Condensed stiffness:
K_el = K_uu - K_uα * (K_αα)^(-1) * K_αu
```

This preserves standard element connectivity while capturing enhanced strain behavior.

### Nine Enhanced Modes for Hex8 (Wilson-Taylor Reference)

The classical Wilson element adds 9 internal displacement modes (3 per direction):
```
Enhanced displacement gradient:
∂u_i/∂x_j += Σ_k [J(0)/J(ξ)] * α_i^(k) * δ_km * ξ_k * (∂ξ_m/∂x_j)

where:
- J(0) = Jacobian at element center
- J(ξ) = Jacobian at integration point
- α_i^(k) = internal DOFs (unknown)
- ξ_k = natural coordinates
```

This modification requires computing volume average of shape derivatives before stiffness assembly.

### B-matrix Decomposition for SRI

Volumetric projection:
```
m = [1, 1, 1, 0, 0, 0]^T  (volumetric selector)
I_vol = (1/3) * m * m^T   (volumetric projector, 6×6)
I_dev = I - I_vol         (deviatoric projector, 6×6)

B_vol = I_vol * B
B_dev = I_dev * B

Stiffness split:
K_vol = Σ_center B_vol^T * D * B_vol * |J| * w
K_dev = Σ_8pt B_dev^T * D * B_dev * |J| * w
K = K_vol + K_dev
```

## References

1. Hughes, T.J.R. "The Finite Element Method" (2000), Ch. 4
2. Simo, J.C. and Rifai, M.S. "A class of mixed assumed strain methods and the method of incompatible modes" (1990), Int J Numer Meth Eng 29:1595-1638
3. Belytschko, T. et al. "Nonlinear Finite Elements for Continua and Structures" (2014)
4. Wilson, E.L., Taylor, R.L., Doherty, W.P., Ghaboussi, J. "Incompatible displacement models" (1973), In: Fenves SJ (ed) Numerical and computer methods in structural mechanics. Academic Press, pp 43-57
5. Simo, J.C., Armero, F., Taylor, R.L. "Improved versions of assumed enhanced strain tri-linear elements for 3D finite deformation problems" (1993), Comput Methods Appl Mech Eng 110:359-386
6. NAFEMS Benchmark Challenge: LE10 Thick Plate
7. ABAQUS Theory Manual: Section 4.5 (Solid Elements)
8. A.F. Bower "Applied Mechanics of Solids", Ch. 8.6 - Special Elements (http://solidmechanics.org/Text/Chapter8_6/Chapter8_6.php)

## Open Questions

1. Should SRI be the default for Hex8, or require explicit selection?
   - **Recommendation:** Keep full integration as default for safety; SRI should be opt-in via Hex8SRI
2. Need stress recovery strategy for SRI (stress at 1 point or extrapolate from 8)?
   - **Recommendation:** Compute stresses at all 8 Gauss points, same as full integration
3. How to handle mixed meshes (Hex8 + Hex8SRI in same model)?
   - **Recommendation:** Support mixed meshes; stiffness assembly treats each element type independently
