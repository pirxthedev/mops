#!/usr/bin/env python3
"""Simple Tensile Bar - Basic Stress/Strain Example.

This example demonstrates the fundamental relationship between
applied load, stress, and strain using a simple bar under tension.

The problem demonstrates:
- Creating a structured hex8 mesh
- Applying boundary conditions and point loads
- Comparing FEA results to analytical solutions

Analytical Solution:
    Stress: sigma = F / A = P / (W * H)
    Strain: epsilon = sigma / E
    Elongation: delta = epsilon * L = P * L / (E * A)

This is the simplest FEA problem and serves as a validation test.

Run with:
    cd mops && pip install -e ./mops-python
    python examples/simple_tensile_bar.py
"""

import numpy as np

from mops import (
    Elements,
    Force,
    Material,
    Mesh,
    Model,
    Nodes,
    solve,
)


def generate_hex8_bar_mesh(
    length: float,
    width: float,
    height: float,
    nx: int,
    ny: int,
    nz: int,
) -> Mesh:
    """Generate a structured hex8 mesh for a rectangular bar.

    Args:
        length: Bar length in x-direction.
        width: Bar width in y-direction.
        height: Bar height in z-direction.
        nx: Number of elements in x-direction.
        ny: Number of elements in y-direction.
        nz: Number of elements in z-direction.

    Returns:
        Hex8 mesh for the bar.
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
                # Hex8 node ordering
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


def main():
    print("=" * 60)
    print("MOPS Example: Simple Tensile Bar")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Problem Setup
    # -------------------------------------------------------------------------
    LENGTH = 100.0   # mm
    WIDTH = 10.0     # mm
    HEIGHT = 10.0    # mm

    # Material properties (aluminum)
    E = 70e3         # MPa (70 GPa)
    NU = 0.33

    # Applied load
    TOTAL_FORCE = 10000.0  # N (10 kN)

    print("Problem Parameters:")
    print(f"  Bar dimensions: {LENGTH} x {WIDTH} x {HEIGHT} mm")
    print(f"  Cross-section area: {WIDTH * HEIGHT} mm^2")
    print(f"  Material: Aluminum (E={E/1e3:.0f} GPa, nu={NU})")
    print(f"  Applied force: {TOTAL_FORCE} N")
    print()

    # -------------------------------------------------------------------------
    # Analytical Solution
    # -------------------------------------------------------------------------
    AREA = WIDTH * HEIGHT  # mm^2

    analytical_stress = TOTAL_FORCE / AREA  # MPa
    analytical_strain = analytical_stress / E
    analytical_elongation = analytical_strain * LENGTH  # mm

    print("Analytical Solution:")
    print(f"  Stress: sigma = F/A = {analytical_stress:.2f} MPa")
    print(f"  Strain: epsilon = sigma/E = {analytical_strain:.6e}")
    print(f"  Elongation: delta = epsilon*L = {analytical_elongation:.6e} mm")
    print()

    # -------------------------------------------------------------------------
    # Create Mesh
    # -------------------------------------------------------------------------
    print("Creating hex8 mesh (10x2x2 elements)...")
    mesh = generate_hex8_bar_mesh(
        LENGTH, WIDTH, HEIGHT,
        nx=10, ny=2, nz=2,
    )
    print(f"  Nodes: {mesh.n_nodes}")
    print(f"  Elements: {mesh.n_elements}")
    print()

    # -------------------------------------------------------------------------
    # Define Material and Model
    # -------------------------------------------------------------------------
    aluminum = Material("aluminum", e=E, nu=NU)

    # Count nodes at loaded end to distribute force
    loaded_end_nodes = Nodes.where(x=LENGTH).evaluate(mesh)
    n_loaded_nodes = len(loaded_end_nodes)
    force_per_node = TOTAL_FORCE / n_loaded_nodes

    print("Building model...")
    print(f"  Fixed end nodes (x=0): {len(Nodes.where(x=0).evaluate(mesh))}")
    print(f"  Loaded end nodes (x={LENGTH}): {n_loaded_nodes}")
    print(f"  Force per node: {force_per_node:.2f} N")

    model = (
        Model(mesh, materials={"aluminum": aluminum})
        .assign(Elements.all(), material="aluminum")
        .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
        .load(Nodes.where(x=LENGTH), Force(fx=force_per_node))
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

    # Get average x-displacement at loaded end
    loaded_x_displacements = [disp[n, 0] for n in loaded_end_nodes]
    avg_elongation = np.mean(loaded_x_displacements)

    # Get stress
    stress = results.stress()
    # Average sigma_xx (stress component 0)
    avg_stress = np.mean(stress[:, 0])

    # Get von Mises stress
    von_mises = results.von_mises()
    avg_von_mises = np.mean(von_mises)

    print("FEA Results:")
    print(f"  Max displacement: {results.max_displacement():.6e} mm")
    print(f"  Avg elongation (x-disp at end): {avg_elongation:.6e} mm")
    print(f"  Avg stress (sigma_xx): {avg_stress:.2f} MPa")
    print(f"  Avg von Mises stress: {avg_von_mises:.2f} MPa")
    print()

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    print("Comparison with Analytical:")
    print(f"  Elongation - FEA: {avg_elongation:.6e}, Analytical: {analytical_elongation:.6e}")
    elongation_error = abs(avg_elongation - analytical_elongation) / analytical_elongation
    print(f"  Elongation error: {elongation_error*100:.2f}%")
    print()

    print(f"  Stress - FEA: {avg_stress:.2f}, Analytical: {analytical_stress:.2f}")
    stress_error = abs(avg_stress - analytical_stress) / analytical_stress
    print(f"  Stress error: {stress_error*100:.2f}%")
    print()

    # -------------------------------------------------------------------------
    # Verify Boundary Conditions
    # -------------------------------------------------------------------------
    fixed_nodes = Nodes.where(x=0).evaluate(mesh)
    max_fixed_disp = max(np.abs(disp[n]).max() for n in fixed_nodes)

    print("Boundary Condition Check:")
    print(f"  Max displacement at fixed end: {max_fixed_disp:.2e}")
    if max_fixed_disp < 1e-10:
        print("  Fixed end properly constrained!")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    if elongation_error < 0.01 and stress_error < 0.05:
        print("SUCCESS: FEA results match analytical solution!")
        print("  (This validates the solver implementation)")
    else:
        print("Results show some deviation from analytical solution.")
        print("  (This is normal for 3D elements due to Poisson effects)")
    print("=" * 60)


if __name__ == "__main__":
    main()
