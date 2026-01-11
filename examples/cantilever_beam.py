#!/usr/bin/env python3
"""Cantilever Beam Example - Classical FEA Benchmark.

This example demonstrates the MOPS API with a classic structural mechanics
problem: a cantilever beam with a tip load.

The beam is fixed at one end (x=0) and loaded at the free end (x=L).
Results are compared against Euler-Bernoulli beam theory.

Analytical Solution:
    Tip deflection: delta = P * L^3 / (3 * E * I)
    where I = b * h^3 / 12 (second moment of area)

Run with:
    cd mops && pip install -e ./mops-python
    python examples/cantilever_beam.py
"""

import numpy as np

from mops import (
    Elements,
    Force,
    Material,
    Mesh,
    Model,
    Nodes,
    Results,
    solve,
)


def generate_hex8_beam_mesh(
    length: float,
    width: float,
    height: float,
    nx: int,
    ny: int,
    nz: int,
) -> Mesh:
    """Generate a structured hex8 mesh for a rectangular beam.

    The beam extends along the x-axis from x=0 (fixed end) to x=length (free end).

    Args:
        length: Beam length in x-direction.
        width: Beam width in y-direction.
        height: Beam height in z-direction.
        nx: Number of elements in x-direction.
        ny: Number of elements in y-direction.
        nz: Number of elements in z-direction.

    Returns:
        Mesh: A hex8 mesh with (nx+1)*(ny+1)*(nz+1) nodes.
    """
    # Generate node coordinates
    x_coords = np.linspace(0, length, nx + 1)
    y_coords = np.linspace(0, width, ny + 1)
    z_coords = np.linspace(0, height, nz + 1)

    nodes = []
    for z in z_coords:
        for y in y_coords:
            for x in x_coords:
                nodes.append([x, y, z])
    nodes = np.array(nodes, dtype=np.float64)

    # Generate element connectivity
    def node_index(i, j, k):
        return i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Hex8 node ordering (counterclockwise from bottom-front-left)
                n0 = node_index(i, j, k)
                n1 = node_index(i + 1, j, k)
                n2 = node_index(i + 1, j + 1, k)
                n3 = node_index(i, j + 1, k)
                n4 = node_index(i, j, k + 1)
                n5 = node_index(i + 1, j, k + 1)
                n6 = node_index(i + 1, j + 1, k + 1)
                n7 = node_index(i, j + 1, k + 1)
                elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

    elements = np.array(elements, dtype=np.int64)
    return Mesh(nodes, elements, "hex8")


def analytical_tip_deflection(P: float, L: float, E: float, I: float) -> float:
    """Compute analytical tip deflection for cantilever with end load.

    Args:
        P: Applied load at tip (force, positive = downward).
        L: Beam length.
        E: Young's modulus.
        I: Second moment of area.

    Returns:
        Tip deflection magnitude.
    """
    return abs(P) * L**3 / (3 * E * I)


def main():
    print("=" * 60)
    print("MOPS Example: Cantilever Beam")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Problem Setup
    # -------------------------------------------------------------------------
    # Geometry (slender beam)
    LENGTH = 10.0  # m
    WIDTH = 1.0    # m
    HEIGHT = 1.0   # m

    # Material properties (steel)
    E = 200e9      # Pa (200 GPa)
    NU = 0.3

    # Applied load (total, distributed across tip nodes)
    TOTAL_LOAD = -1000.0  # N (negative = downward in z)

    print("Problem Parameters:")
    print(f"  Beam dimensions: {LENGTH} x {WIDTH} x {HEIGHT} m")
    print(f"  Material: Steel (E={E/1e9:.0f} GPa, nu={NU})")
    print(f"  Total tip load: {TOTAL_LOAD} N")
    print()

    # -------------------------------------------------------------------------
    # Analytical Solution
    # -------------------------------------------------------------------------
    I = WIDTH * HEIGHT**3 / 12  # Second moment of area
    analytical_delta = analytical_tip_deflection(abs(TOTAL_LOAD), LENGTH, E, I)
    print(f"Analytical Solution (Euler-Bernoulli beam theory):")
    print(f"  I = {I:.6f} m^4")
    print(f"  Tip deflection = {analytical_delta:.6e} m")
    print()

    # -------------------------------------------------------------------------
    # Create Mesh
    # -------------------------------------------------------------------------
    print("Creating hex8 mesh (20x4x4 elements)...")
    mesh = generate_hex8_beam_mesh(
        LENGTH, WIDTH, HEIGHT,
        nx=20, ny=4, nz=4,
    )
    print(f"  Nodes: {mesh.n_nodes}")
    print(f"  Elements: {mesh.n_elements}")
    print()

    # -------------------------------------------------------------------------
    # Define Material
    # -------------------------------------------------------------------------
    steel = Material("steel", e=E, nu=NU)

    # -------------------------------------------------------------------------
    # Build Model
    # -------------------------------------------------------------------------
    print("Building model...")

    # Count tip nodes to distribute load
    tip_nodes = Nodes.where(x=LENGTH).evaluate(mesh)
    n_tip_nodes = len(tip_nodes)
    per_node_load = TOTAL_LOAD / n_tip_nodes

    print(f"  Tip nodes: {n_tip_nodes}")
    print(f"  Load per node: {per_node_load:.2f} N")

    model = (
        Model(mesh, materials={"steel": steel})
        .assign(Elements.all(), material="steel")
        .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
        .load(Nodes.where(x=LENGTH), Force(fz=per_node_load))
    )
    print("  Model built successfully!")
    print()

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    print("Solving...")
    results = solve(model)
    print("  Solve complete!")
    print()

    # -------------------------------------------------------------------------
    # Extract Results
    # -------------------------------------------------------------------------
    disp = results.displacement()

    # Get tip displacements
    tip_z_displacements = [disp[n, 2] for n in tip_nodes]
    avg_tip_deflection = np.mean(tip_z_displacements)

    print("Results:")
    print(f"  Max displacement: {results.max_displacement():.6e} m")
    print(f"  Avg tip z-displacement: {avg_tip_deflection:.6e} m")
    print()

    # -------------------------------------------------------------------------
    # Comparison with Analytical
    # -------------------------------------------------------------------------
    fea_deflection = abs(avg_tip_deflection)
    relative_error = abs(fea_deflection - analytical_delta) / analytical_delta

    print("Comparison with Beam Theory:")
    print(f"  FEA tip deflection:       {fea_deflection:.6e} m")
    print(f"  Analytical tip deflection: {analytical_delta:.6e} m")
    print(f"  Relative error: {relative_error*100:.2f}%")
    print()

    # -------------------------------------------------------------------------
    # Verify Fixed End
    # -------------------------------------------------------------------------
    fixed_nodes = Nodes.where(x=0).evaluate(mesh)
    max_fixed_disp = max(np.abs(disp[n]).max() for n in fixed_nodes)

    print("Boundary Condition Check:")
    print(f"  Max displacement at fixed end: {max_fixed_disp:.2e} m")
    if max_fixed_disp < 1e-10:
        print("  Fixed end properly constrained!")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    if relative_error < 0.30:
        print("SUCCESS: FEA results within 30% of analytical solution.")
    else:
        print("NOTE: Larger error expected with coarse mesh or short beams.")
    print("=" * 60)


if __name__ == "__main__":
    main()
