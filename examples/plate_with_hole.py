#!/usr/bin/env python3
"""Plate with Circular Hole - Stress Concentration Example.

This example demonstrates stress concentration analysis using MOPS.
A plate with a circular hole is subjected to uniaxial tension.

The problem demonstrates:
- 3D mesh generation for a plate with cutout
- Applying tension loads
- Analyzing stress concentration

Analytical Solution (Kirsch Solution for infinite plate):
    Kt = 3.0 (stress concentration factor at hole edge)
    Maximum stress = Kt * applied stress = 3 * sigma

For finite plate geometry, the stress concentration is slightly different.

Run with:
    cd mops && pip install -e ./mops-python
    python examples/plate_with_hole.py
"""

import math
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


def generate_plate_with_hole_mesh(
    plate_length: float,
    plate_width: float,
    thickness: float,
    hole_radius: float,
    n_radial: int = 8,
    n_angular: int = 16,
    n_thick: int = 2,
) -> Mesh:
    """Generate hex8 mesh for quarter plate with circular hole.

    Uses quarter symmetry - only models x >= 0, y >= 0 quadrant.
    The hole is centered at the origin.

    Args:
        plate_length: Plate half-length in x-direction (from center to edge).
        plate_width: Plate half-width in y-direction (from center to edge).
        thickness: Plate thickness in z-direction.
        hole_radius: Radius of the circular hole.
        n_radial: Number of elements from hole to outer edge.
        n_angular: Number of elements around the quarter arc.
        n_thick: Number of elements through thickness.

    Returns:
        Hex8 mesh for the quarter plate.
    """
    nodes = []
    elements = []

    # Radial positions (from hole to plate edge)
    # Use grading to concentrate elements near hole
    r_params = np.linspace(0, 1, n_radial + 1) ** 1.5

    # Angular positions (0 to pi/2 for quarter)
    angles = np.linspace(0, math.pi / 2, n_angular + 1)

    # Thickness positions
    z_coords = np.linspace(0, thickness, n_thick + 1)

    # Create a map for node indices
    node_map = {}

    for i_z, z in enumerate(z_coords):
        for i_r, r_param in enumerate(r_params):
            for i_theta, theta in enumerate(angles):
                # Map r_param to actual radius
                # At r_param=0: r=hole_radius, at r_param=1: interpolate to plate edge
                r_at_hole = hole_radius

                # Outer boundary depends on angle (rectangular plate)
                if abs(math.cos(theta)) > 0.01:
                    r_x = plate_length / math.cos(theta)
                else:
                    r_x = float('inf')

                if abs(math.sin(theta)) > 0.01:
                    r_y = plate_width / math.sin(theta)
                else:
                    r_y = float('inf')

                r_outer = min(r_x, r_y)
                r = r_at_hole + r_param * (r_outer - r_at_hole)

                x = r * math.cos(theta)
                y = r * math.sin(theta)

                node_idx = len(nodes)
                nodes.append([x, y, z])
                node_map[(i_r, i_theta, i_z)] = node_idx

    # Generate hex8 elements
    for i_z in range(n_thick):
        for i_r in range(n_radial):
            for i_theta in range(n_angular):
                # Bottom face
                n0 = node_map[(i_r, i_theta, i_z)]
                n1 = node_map[(i_r + 1, i_theta, i_z)]
                n2 = node_map[(i_r + 1, i_theta + 1, i_z)]
                n3 = node_map[(i_r, i_theta + 1, i_z)]

                # Top face
                n4 = node_map[(i_r, i_theta, i_z + 1)]
                n5 = node_map[(i_r + 1, i_theta, i_z + 1)]
                n6 = node_map[(i_r + 1, i_theta + 1, i_z + 1)]
                n7 = node_map[(i_r, i_theta + 1, i_z + 1)]

                elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

    nodes_array = np.array(nodes, dtype=np.float64)
    elements_array = np.array(elements, dtype=np.int64)

    return Mesh(nodes_array, elements_array, "hex8")


def get_element_centroids(mesh: Mesh) -> np.ndarray:
    """Compute centroids of all elements.

    Args:
        mesh: The mesh.

    Returns:
        (n_elements, 3) array of element centroid coordinates.
    """
    coords = mesh.coords
    elements = mesh.elements
    centroids = []

    for elem in elements:
        elem_coords = coords[elem]
        centroid = np.mean(elem_coords, axis=0)
        centroids.append(centroid)

    return np.array(centroids)


def main():
    print("=" * 60)
    print("MOPS Example: Plate with Circular Hole")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Problem Setup
    # -------------------------------------------------------------------------
    # Geometry (quarter model)
    PLATE_LENGTH = 100.0  # mm (half-length in x)
    PLATE_WIDTH = 100.0   # mm (half-width in y)
    THICKNESS = 10.0      # mm
    HOLE_RADIUS = 10.0    # mm

    # Material properties (steel)
    E = 210e3   # MPa (210 GPa)
    NU = 0.3

    # Applied stress (tension in x-direction)
    APPLIED_STRESS = 100.0  # MPa

    print("Problem Parameters:")
    print(f"  Plate dimensions: {2*PLATE_LENGTH} x {2*PLATE_WIDTH} x {THICKNESS} mm")
    print(f"  Hole radius: {HOLE_RADIUS} mm")
    print(f"  Material: Steel (E={E/1e3:.0f} GPa, nu={NU})")
    print(f"  Applied tension: {APPLIED_STRESS} MPa")
    print()

    # Analytical reference (Kirsch solution for infinite plate)
    Kt_analytical = 3.0
    max_stress_analytical = Kt_analytical * APPLIED_STRESS

    print("Analytical Reference (Kirsch solution for infinite plate):")
    print(f"  Stress concentration factor Kt = {Kt_analytical}")
    print(f"  Maximum stress at hole edge = {max_stress_analytical} MPa")
    print("  Note: Finite plate will have slightly different Kt")
    print()

    # -------------------------------------------------------------------------
    # Create Mesh
    # -------------------------------------------------------------------------
    print("Creating hex8 mesh for quarter plate...")
    mesh = generate_plate_with_hole_mesh(
        PLATE_LENGTH, PLATE_WIDTH, THICKNESS, HOLE_RADIUS,
        n_radial=12, n_angular=24, n_thick=2,
    )
    print(f"  Nodes: {mesh.n_nodes}")
    print(f"  Elements: {mesh.n_elements}")
    print()

    # -------------------------------------------------------------------------
    # Define Material
    # -------------------------------------------------------------------------
    steel = Material("steel", e=E, nu=NU)

    # -------------------------------------------------------------------------
    # Apply Boundary Conditions and Loads
    # -------------------------------------------------------------------------
    print("Building model...")

    coords = mesh.coords

    # Find nodes on inner hole boundary to fix (as approximation of symmetry)
    hole_nodes = []
    for i, (x, y, z) in enumerate(coords):
        r = math.sqrt(x**2 + y**2)
        if abs(r - HOLE_RADIUS) < 0.5:  # tolerance
            hole_nodes.append(i)

    # Find nodes on outer x-edge for loading
    outer_x_nodes = [i for i, (x, y, z) in enumerate(coords)
                     if abs(x - PLATE_LENGTH) < 0.5]

    n_outer = len(outer_x_nodes)
    print(f"  Nodes on hole boundary: {len(hole_nodes)}")
    print(f"  Nodes on outer x-edge: {n_outer}")

    # Calculate force per node
    # Total force = stress * area = stress * (width * thickness)
    # For quarter model: area = PLATE_WIDTH * THICKNESS
    total_force = APPLIED_STRESS * PLATE_WIDTH * THICKNESS  # N
    force_per_node = total_force / n_outer

    print(f"  Total force: {total_force:.0f} N")
    print(f"  Force per node: {force_per_node:.2f} N")

    # Build model
    # Fix hole boundary to simulate rigid inclusion (simplified BC)
    # Apply tension on outer edge
    model = (
        Model(mesh, materials={"steel": steel})
        .assign(Elements.all(), material="steel")
        .constrain(Nodes.by_indices(hole_nodes), dofs=["ux", "uy", "uz"])
        .load(Nodes.by_indices(outer_x_nodes), Force(fx=force_per_node))
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
    print("Results:")
    print(f"  Max displacement: {results.max_displacement():.6e} mm")

    # Get stresses
    von_mises = results.von_mises()
    stress = results.stress()

    max_vm = np.max(von_mises)
    print(f"  Max von Mises stress: {max_vm:.1f} MPa")

    # Get element centroids for analysis
    centroids = get_element_centroids(mesh)

    # Find elements near hole (at r ~ hole_radius)
    hole_elements = []
    for i, centroid in enumerate(centroids):
        r = math.sqrt(centroid[0]**2 + centroid[1]**2)
        # Elements with centroid between hole and 1.5x hole radius
        if HOLE_RADIUS < r < HOLE_RADIUS * 1.5:
            hole_elements.append(i)

    print(f"  Elements near hole edge: {len(hole_elements)}")

    if hole_elements:
        vm_near_hole = [von_mises[e] for e in hole_elements]
        max_vm_hole = max(vm_near_hole)
        avg_vm_hole = np.mean(vm_near_hole)

        # Stress concentration based on von Mises
        Kt_fea = max_vm_hole / APPLIED_STRESS

        print()
        print("Stress Near Hole Edge:")
        print(f"  Max von Mises: {max_vm_hole:.1f} MPa")
        print(f"  Avg von Mises: {avg_vm_hole:.1f} MPa")
        print(f"  Stress concentration Kt (FEA): {Kt_fea:.2f}")
    else:
        Kt_fea = max_vm / APPLIED_STRESS

    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Comparison with Analytical (Kirsch solution):")
    print(f"  Analytical Kt: {Kt_analytical}")
    print(f"  FEA Kt: {Kt_fea:.2f}")
    print()
    print("Note: Results use simplified BCs (rigid hole instead of symmetry).")
    print("This demonstrates stress concentration but differs from true Kt.")
    print("=" * 60)


if __name__ == "__main__":
    main()
