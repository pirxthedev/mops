# FEA Conformance Test Index

Master catalog of all tests for the FEA conformance test suite. See [2026-01-13-fea-conformance-spec.md](2026-01-13-fea-conformance-spec.md) for the test suite design and format specifications.

---

## NAFEMS

Tests from NAFEMS benchmark publications (R0011, R0015, R0016, etc.)

### Linear Elastic Tests (from "The Standard NAFEMS Benchmarks" TNSB Rev. 3)

| Test ID | Name | Description | Target Value |
|---------|------|-------------|--------------|
| LE1 | Elliptic Membrane | Plane stress elliptic membrane under edge pressure | σ_yy at point D |
| LE2 | Cylindrical Shell Patch Test | Cylindrical shell bending patch test | Constant strain verification |
| LE3 | Hemisphere with Point Loads | Quarter hemisphere with radial point loads | u_x at point A = 0.0924 |
| LE4 | Axisymmetric Hyperbolic Shell | Hyperbolic shell under axisymmetric loading | Stress at specified location |
| LE5 | Z-Section Cantilever | Thin-walled Z-section beam under torque | σ_xx at point P |
| LE6 | Skew Plate Under Normal Pressure | Rhombic plate with simply supported edges | σ_1 at center (lower surface) |
| LE7 | Axisymmetric Cylinder/Sphere | Axisymmetric pressure loading | Stress distribution |
| LE8 | Axisymmetric Shell with Pressure | Axisymmetric shell under pressure | Stress verification |
| LE9 | Axisymmetric Branched Shell | Shell intersection geometry | Junction stresses |
| LE10 | Thick Plate Under Pressure | Thick plate with pressure loading | σ_yy at specified location |
| LE11 | Solid Cylinder/Taper/Sphere - Temperature | Thermal loading on composite geometry | Thermal stress distribution |

### Thermal Tests

| Test ID | Name | Description | Target Value |
|---------|------|-------------|--------------|
| T1 | Membrane with Hotspot | 2D thermal with localized heat source | Temperature at hotspot |
| T2 | 1D Heat Transfer with Radiation | Radiation boundary condition test | Temperature profile |
| T3 | 1D Transient Heat Transfer | Time-dependent thermal conduction | Temperature vs. time |
| T4 | 2D Heat Transfer with Convection | Convective boundary condition test | Temperature distribution |

### Free Vibration Tests (from R0015 "Selected Benchmarks for Natural Frequency Analysis")

| Test ID | Name | Description | Target Frequency |
|---------|------|-------------|------------------|
| FV1 | Free Thin Square Plate | FFFF boundary conditions | Multiple modes |
| FV2 | Clamped Thin Rhombic Plate | CCCC skew plate | Fundamental frequency |
| FV3 | Cantilevered Thin Square Plate | CFFF plate | Bending modes |
| FV4 | Thick Square Plate | Mindlin plate theory | Shear effects |
| FV5 | Pin-ended Double Cross | Beam structure | In-plane vibration |
| FV6 | Cantilevered Tapered Membrane | Variable thickness | Mode shapes |
| FV7 | Free Cylinder | Cylindrical shell | Circumferential modes |
| FV8 | Thick Hollow Sphere | 3D solid | Breathing mode |
| FV9 | Cantilever with Off-center Point Mass | Mass modification | Frequency shift |
| FV10 | Deep Solid Beam | Timoshenko beam | Shear correction |
| FV11 | Solid Square Plate | 3D plate | Through-thickness modes |
| FV12 | Deep Simply-supported Beam | Thick beam | Higher modes |

### Forced Vibration Tests (from R0016)

| Test ID | Name | Description | Target Response |
|---------|------|-------------|-----------------|
| FFV1-3 | Forced vibration benchmarks | Dynamic response under harmonic loading | Displacement amplitude |

### Nonlinear Benchmarks

| Test ID | Name | Description | Target Value |
|---------|------|-------------|--------------|
| NL1 | Geometric nonlinearity tests | Large displacement/rotation | Equilibrium path |
| NL2 | Material nonlinearity tests | Plasticity verification | Yield behavior |

---

## MacNeal-Harder

Element accuracy tests from MacNeal and Harder's landmark paper (Finite Elements in Analysis and Design, 1985).

### Patch Tests

| Test | Description | Element Types | Verification |
|------|-------------|---------------|--------------|
| Membrane Patch Test | Arbitrary distorted mesh under constant strain | Plate/shell | Exact stress reproduction |
| Bending Plate Patch Test | Distorted mesh under constant curvature | Plate/shell | Exact moment reproduction |
| Solid Patch Test | 3D elements under constant strain | Brick/solid | Stress accuracy |

### Beam Problems

| Test | Description | Reference Solution |
|------|-------------|--------------------|
| Straight Cantilever Beam | End-loaded cantilever, thin/thick cases | Timoshenko beam theory |
| Curved Beam | 90° arc cantilever with end load | Analytical (in-plane, out-of-plane) |
| Twisted Beam | 90° pre-twisted cantilever | In-plane tip: 0.005424, Out-of-plane tip: 0.001754 |

### Shell Problems

| Test | Description | Reference Solution |
|------|-------------|--------------------|
| Scordelis-Lo Roof | Cylindrical roof segment under gravity | u_z = 0.3024 (at free edge midpoint) |
| Pinched Hemisphere | Open hemisphere with point loads | u_x = 0.0924 (at load point) |
| Pinched Cylinder | Cylinder with diaphragm ends, point loads | u_z = 1.8248×10⁻⁵ |

### Solid Problems

| Test | Description | Reference Solution |
|------|-------------|--------------------|
| Thick-walled Cylinder | Nearly incompressible, internal pressure | Lamé solution |

---

## Patch Tests

Constant strain reproduction tests for element validation (originated by Bruce Irons).

### Standard Patch Tests

| Test Type | Purpose | Pass Criteria |
|-----------|---------|---------------|
| Displacement Patch Test | Verify constant strain reproduction | Exact nodal displacements |
| Force Patch Test | Verify equilibrium | Correct nodal forces |
| Higher-Order Patch Test | Linear strain field reproduction | Quadratic displacement accuracy |

### Specific Element Patch Tests

| Element Type | Loading | Verification |
|--------------|---------|--------------|
| Membrane elements | Uniform tension, shear | Constant stress field |
| Plate bending elements | Constant curvature | Uniform moment field |
| Shell elements | Combined membrane/bending | Both stress types |
| 3D solid elements | Uniaxial, biaxial, triaxial strain | Stress tensor accuracy |
| Incompatible/nonconforming elements | Weak patch test | Convergence to correct solution |

### Patch Test Geometry

Standard configuration:

- Rectangular exterior boundary
- Irregular internal node placement
- At least one fully internal node
- Distorted element shapes

---

## Shell Classics

Classic shell benchmark problems (Scordelis-Lo, Raasch hook, pinched cylinder).

### Shell Obstacle Course (Belytschko et al.)

| Test | Geometry | Loading | Reference Value | Tests |
|------|----------|---------|-----------------|-------|
| Scordelis-Lo Roof | Cylindrical segment, L=50, R=25, θ=40°, t=0.25 | Gravity (90 N/m²) | u_z = 0.3024 (shear-flexible) or 0.3006 (shear-rigid) | Membrane/bending coupling |
| Pinched Hemisphere | R=10, t=0.04, 18° hole at apex | Concentrated radial loads (±2) | u_x = 0.0924 | Inextensional bending |
| Pinched Cylinder | L=600, R=300, t=3, rigid diaphragms | Point loads at center | u_z = 1.8248×10⁻⁵ | Complex membrane states |

### Additional Shell Benchmarks

| Test | Description | Key Feature |
|------|-------------|-------------|
| Raasch Hook | Curved strip hook with in-plane shear load at tip | Bending-extension-twist coupling |
| Twisted Beam (shell model) | Pre-twisted cantilever | Warped geometry |
| Hemispherical Shell with Hole | Point-loaded hemisphere | Double curvature |
| Cylindrical Panel | Hinged cylindrical panel | Snap-through buckling |

### Shell Test Parameters

**Scordelis-Lo Roof:**

- E = 4.32×10⁸, ν = 0.0
- Rigid diaphragms at curved ends
- Free longitudinal edges
- Target: vertical displacement at midpoint of free edge

**Raasch Hook (BMW challenge, 1990):**

- Two cylindrical strips with different radii
- Clamped at one end
- Tip shear load
- Target: tip displacement ≈ 5.02 (consensus value)

---

## Manufactured Solutions

Method of manufactured solutions for convergence verification.

### Methodology

1. Choose smooth manufactured solution u_mfg(x,y,z,t)
2. Substitute into governing PDE to obtain source term f
3. Solve modified problem with source term
4. Compare numerical solution to u_mfg
5. Verify expected convergence rate with mesh refinement

### Standard Test Functions

| Problem Type | Example Manufactured Solution | Expected Order |
|--------------|-------------------------------|----------------|
| Poisson equation | sin(2πx)sin(2πy) | p+1 (L² norm) |
| Elasticity | Trigonometric displacement field | 2 (displacement), 1 (stress) |
| Heat conduction | Polynomial in space, exponential in time | Depends on scheme |
| Navier-Stokes | Smooth velocity/pressure pair | Varies by method |

### MMS Best Practices

- Solution should exercise all terms in PDE
- Boundary conditions must match manufactured solution
- Avoid solutions exactly representable by shape functions
- Use smooth functions (sin, cos, polynomials)
- Spatial convergence: hold time step small
- Temporal convergence: hold mesh fine

### Verification Metrics

| Metric | Definition | Use |
|--------|------------|-----|
| L² error norm | √(∫(u-u_h)² dΩ) | Global accuracy |
| L∞ error norm | max\|u-u_h\| | Point accuracy |
| H¹ error semi-norm | √(∫\|∇(u-u_h)\|² dΩ) | Gradient accuracy |
| Observed order of convergence | log(e₁/e₂)/log(h₁/h₂) | Rate verification |

---

## Textbook

Problems from Timoshenko, Roark's Formulas, and other standard references.

### Timoshenko Beam Solutions

| Problem | Configuration | Analytical Solution |
|---------|---------------|---------------------|
| Cantilever with end load | Euler-Bernoulli or Timoshenko | δ = PL³/3EI (+ shear correction) |
| Simply supported with central load | Point load at midspan | δ_max = PL³/48EI |
| Fixed-fixed with uniform load | Clamped ends | δ_max = wL⁴/384EI |
| Cantilever with distributed load | Uniform load | δ_tip = wL⁴/8EI |

### Timoshenko-Goodier 2D Elasticity

| Problem | Description | Solution Type |
|---------|-------------|---------------|
| Cantilever beam (plane stress) | End load, parabolic shear traction | Closed-form stress field |
| Plate with circular hole | Uniaxial tension | Stress concentration (Kt=3) |
| Hertzian contact | Cylinder on half-space | Contact pressure distribution |
| Wedge under tip load | Concentrated force at apex | Stress singularity |

### Roark's Formulas Cases

| Category | Example Problems |
|----------|-----------------|
| Beams | All standard boundary conditions, concentrated/distributed loads |
| Curved beams | Circular rings, hooks |
| Flat plates | Circular and rectangular, various supports |
| Shells of revolution | Cylinders, spheres, cones |
| Pressure vessels | Thin/thick wall, end effects |
| Torsion | Circular, non-circular cross-sections |

### Lamé Solutions (Thick-Walled Cylinders/Spheres)

| Configuration | Stress Formulas |
|---------------|-----------------|
| Internal pressure only | σ_r = -p_i(b²/r² - 1)/(b²/a² - 1) |
| External pressure only | σ_r = p_o(b²/r² - 1)/(b²/a² - 1) |
| Combined pressure | Superposition of above |
| Thick sphere | Similar form with r³ terms |

### Kirsch Solution (Plate with Hole)

- Uniaxial tension σ_0 at infinity
- Circular hole of radius a
- Maximum stress at hole edge: σ_max = 3σ_0
- Stress concentration factor: K_t = 3.0

### Additional Textbook Problems

| Source | Problem Type |
|--------|--------------|
| Theory of Elasticity (Timoshenko & Goodier) | 2D/3D analytical solutions |
| Strength of Materials (Timoshenko) | Beam, column, torsion |
| Plates and Shells (Timoshenko & Woinowsky-Krieger) | Plate bending, shell theory |
| Peterson's Stress Concentration Factors | K_t charts for various geometries |

---

## Code_Aster

Curated selection from Code_Aster verification and validation suite.

### Test Case Categories

Code_Aster organizes ~3000+ test cases by prefix:

| Prefix | Category | Description |
|--------|----------|-------------|
| SSLL | Linear Statics - Beams | Beam elements, frames |
| SSLS | Linear Statics - Shells | Shell/plate elements |
| SSLV | Linear Statics - Volumes | 3D solid elements |
| SSNL | Nonlinear Statics | Geometric/material nonlinearity |
| SSNP | Nonlinear Plasticity | Plasticity models |
| SSNV | Nonlinear Various | Mixed nonlinear problems |
| SDLL | Linear Dynamics | Modal analysis |
| SDNL | Nonlinear Dynamics | Transient nonlinear |
| TTLL | Linear Thermal | Steady-state heat |
| TTNL | Nonlinear Thermal | Transient/nonlinear thermal |
| FORMA | Training | Tutorial problems |
| ZZZZ | Miscellaneous | Special tests |

### AFNOR Guide Validation Tests (SSLL series)

| Test ID | Name (French) | Description |
|---------|---------------|-------------|
| SSLL01 | Poutre élancée sur deux appuis encastrés | Slender beam on two clamped supports |
| SSLL02 | Poutre courte sur deux appuis articulés | Short beam on hinged supports |
| SSLL03 | Poutre élancée sur trois appuis | Slender beam on three supports |
| SSLL04 | Structure spatiale rotulée | 3D pinned structure on elastic supports |
| SSLL05 | Bilame | Bimetallic strip |
| SSLL06 | Arc mince encastré (flexion plane) | Clamped thin arc, in-plane bending |
| SSLL07 | Arc mince encastré (flexion hors plan) | Clamped thin arc, out-of-plane |
| SSLL08 | Arc mince bi-articulé | Hinged thin arc |
| SSLL09 | Deux barres à trois rotules | Two bars with three hinges |
| SSLL10 | Portique à liaisons latérales | Portal frame with lateral connections |
| SSLL11 | Treillis de barres articulées | Articulated bar truss |
| SSLL12 | Système triangulé | Triangulated system |
| SSLL13 | Poutre sous-tendue | Suspended beam |
| SSLL14 | Portique plan articulé en pied | Plane portal, pinned at base |
| SSLL15 | Poutre sur sol élastique (libre) | Beam on elastic foundation, free ends |
| SSLL16 | Poutre sur sol élastique (articulé) | Beam on elastic foundation, hinged ends |

### Notable Validation Cases

| Test ID | Description | Reference |
|---------|-------------|-----------|
| NAFEMS cases | Various NAFEMS benchmarks implemented | NAFEMS publications |
| SSLV311 | Murakami crack problem | Fracture mechanics |
| SSNV112 | Contact mechanics | Nonlinear contact |
| SDLL152 | Dynamics with damping | Modal superposition |

### Code_Aster Documentation Structure

- V0-V9: Validation document categories
- Each test includes: problem description, analytical/reference solution, mesh details, results comparison
- Tests available in `/testing/tests` directory of distribution

---

## References

### Primary Sources

1. NAFEMS, "The Standard NAFEMS Benchmarks," TNSB Rev. 3, October 1990
2. NAFEMS, "Selected Benchmarks for Natural Frequency Analysis," R0015, 1987
3. NAFEMS, "Selected Benchmarks for Forced Vibration," R0016, 1989
4. MacNeal, R.H. and Harder, R.L., "A Proposed Standard Set of Problems to Test Finite Element Accuracy," Finite Elements in Analysis and Design, Vol. 1, pp. 3-20, 1985
5. Belytschko, T. et al., "Stress projection for membrane and shear locking in shell finite elements," Computer Methods in Applied Mechanics and Engineering, Vol. 51, pp. 221-258, 1985
6. AFNOR, "Guide de validation des progiciels de calcul de structures," 1990
7. Timoshenko, S. and Goodier, J.N., "Theory of Elasticity," McGraw-Hill
8. Roark, R.J. et al., "Roark's Formulas for Stress and Strain," McGraw-Hill

### Code Verification Standards

- ASME V&V 10-2019: Verification and Validation in Computational Solid Mechanics
- ASME V&V 20-2009: Verification and Validation in Computational Fluid Dynamics
- ASME V&V 40-2018: Assessing Credibility of Computational Modeling for Medical Devices
- AIAA G-077-1998: Guide for Verification and Validation of CFD Simulations

### Software Documentation

- Abaqus Benchmarks Manual
- ANSYS Verification Manual
- DIANA Verification Report
- Code_Aster Validation Documentation
- OptiStruct Verification Manual
