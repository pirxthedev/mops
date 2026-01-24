# FEA Conformance Test Suite Design

## Purpose

An open-source, comprehensive conformance test suite for finite element analysis codes. The suite provides specifications derived from publicly available benchmarks, enabling anyone to validate an FEA implementation or build one from scratch using agentic coding tools.

## Repository Structure

```
fea-conformance/
    README.md
    INDEX.md
    /tests
        /nafems
        /macneal-harder
        /patch-tests
        /shell-classics
        /manufactured
        /textbook
        /code-aster
```

Tests organized by source. Each test is a single markdown file.

## Index Format

`INDEX.md` contains a plain list of links grouped by source, with a one-line description per test:

```markdown
## NAFEMS

- [LE1 Elliptic Membrane](tests/nafems/le1-elliptic-membrane.md) - 2D membrane under pressure, tests plane stress accuracy
- [LE10 Thick Plate](tests/nafems/le10-thick-plate.md) - 3D thick plate in bending, tests solid element stress recovery
```

No metadata table. The index is navigation only.

## Test Case Format

Minimal template, adapting to each source's natural structure:

```markdown
# Test Name

## Source

Citation and original reference.

## Problem Description

Geometry, materials, loads, boundary conditions in prose and tables as appropriate.

## Expected Results

Reference values with tolerances.

## Notes

Our choices, clarifications, known issues. Optional section.
```

## Coverage

### Problem Types

- Linear static structural
- Thermal/heat transfer
- Modal/eigenvalue analysis
- Nonlinear (material, geometric, contact)
- Dynamic/transient

### Element Types

- 1D: Beam (Euler-Bernoulli, Timoshenko), rod/truss
- 2D: Tri3, Tri6, Quad4, Quad8, shell (thin, thick)
- 3D: Tet4, Tet10, Hex8, Hex20

Standard integration variants (full, reduced, selective) where applicable.

### Sources

See [2026-01-24-fea-conformance-test-index.md](2026-01-24-fea-conformance-test-index.md) for the complete test catalog.

- NAFEMS benchmarks (R0011, R0015, R0016, etc.)
- MacNeal-Harder element tests
- Shell classics (Scordelis-Lo, Raasch hook, pinched cylinder)
- Patch tests (constant strain reproduction)
- Method of manufactured solutions (convergence verification)
- Textbook problems (Timoshenko, Roark's)
- Code_Aster V&V (curated selection)

## Tolerances

- Use source-specified tolerances when available
- Default to plus or minus 2% relative error when not specified
- Clearly mark which tolerances are our choices versus source-specified

## Licensing and Attribution

- Repository under MIT or Apache 2.0
- All specifications written in original prose
- Each test case cites its source
- Problems (geometry, loads, physics) are factual and not copyrightable
- Diagrams recreated or described, not copied from sources

## Documentation

Minimal. The repository contains:

- `README.md`: One-paragraph description, license, link to index, brief usage note
- `INDEX.md`: Navigation to all tests
- Test case files: The specifications themselves

No tutorials, implementation guides, or educational material.
