"""Verification tests for cantilever beam against analytical solution.

This module tests the FEA solver against the classical cantilever beam problem
with known closed-form solutions from Euler-Bernoulli beam theory.

Analytical Solution (Euler-Bernoulli beam theory):
-------------------------------------------------
For a cantilever beam with:
- Length L
- Rectangular cross-section: width b, height h
- Young's modulus E
- End load P (applied at free end)

Tip deflection:
    delta = P * L^3 / (3 * E * I)

where I = b * h^3 / 12 is the second moment of area.

Maximum bending stress (at fixed end, top/bottom surface):
    sigma_max = M * c / I = P * L * (h/2) / I = 6 * P * L / (b * h^2)

Reference:
- Timoshenko & Goodier, "Theory of Elasticity" (3rd ed., 1970)
- Roark's Formulas for Stress and Strain (8th ed.)

Notes on FEA vs Beam Theory:
----------------------------
- Beam theory assumes plane sections remain plane and ignores shear deformation
- 3D FEA captures shear effects and Poisson's ratio effects
- For slender beams (L/h > 10), the two should agree within a few percent
- For short beams, FEA will show larger deflection due to shear
- Higher-order elements (Tet10, Hex20) converge faster than linear elements
"""

import numpy as np
import pytest

from mops import (
    Elements,
    Force,
    Material,
    Mesh,
    Model,
    Nodes,
    solve,
)


def generate_hex8_cantilever_mesh(
    length: float,
    width: float,
    height: float,
    nx: int,
    ny: int,
    nz: int,
) -> Mesh:
    """Generate a structured hex8 mesh for a cantilever beam.

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
    for k, z in enumerate(z_coords):
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                nodes.append([x, y, z])
    nodes = np.array(nodes, dtype=np.float64)

    # Generate element connectivity
    # Node numbering: node(i,j,k) = i + j*(nx+1) + k*(nx+1)*(ny+1)
    def node_index(i, j, k):
        return i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Hex8 node ordering (counterclockwise from bottom-front-left):
                # Bottom face: 0,1,2,3 (CCW when viewed from below)
                # Top face: 4,5,6,7 (CCW when viewed from above)
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


def analytical_tip_deflection(
    P: float, L: float, E: float, I: float
) -> float:
    """Compute analytical tip deflection for cantilever with end load.

    Args:
        P: Applied load at tip (force).
        L: Beam length.
        E: Young's modulus.
        I: Second moment of area.

    Returns:
        Tip deflection (same sign as P for downward load).
    """
    return P * L**3 / (3 * E * I)


def analytical_max_stress(
    P: float, L: float, b: float, h: float
) -> float:
    """Compute analytical maximum bending stress at fixed end.

    The maximum stress occurs at the fixed end, at the top/bottom surface.

    Args:
        P: Applied load at tip (magnitude).
        L: Beam length.
        b: Beam width.
        h: Beam height.

    Returns:
        Maximum bending stress (always positive for stress magnitude).
    """
    I = b * h**3 / 12
    c = h / 2  # Distance from neutral axis to surface
    M = abs(P) * L  # Bending moment at fixed end
    return M * c / I


def _apply_distributed_load(mesh: Mesh, model: Model, x_coord: float, total_fz: float) -> Model:
    """Apply a total load distributed equally across nodes at x=x_coord.

    The current API applies Force per-node. To get correct total load,
    we divide by the number of nodes receiving the load.
    """
    tip_nodes = Nodes.where(x=x_coord).evaluate(mesh)
    n_tip_nodes = len(tip_nodes)
    per_node_load = total_fz / n_tip_nodes
    return model.load(Nodes.where(x=x_coord), Force(fz=per_node_load))


class TestCantileverBeamAnalytical:
    """Test cantilever beam against Euler-Bernoulli beam theory."""

    # Beam geometry (slender beam for beam theory validity)
    LENGTH = 10.0  # m
    WIDTH = 1.0    # m
    HEIGHT = 1.0   # m

    # Material properties (steel)
    E = 200e9      # Pa
    NU = 0.3

    # Applied load (total)
    TOTAL_LOAD = -1000.0  # N (negative = downward in z)

    @pytest.fixture
    def steel(self) -> Material:
        """Steel material for testing."""
        return Material("steel", e=self.E, nu=self.NU)

    @pytest.fixture
    def analytical_deflection(self) -> float:
        """Compute expected tip deflection from beam theory."""
        I = self.WIDTH * self.HEIGHT**3 / 12
        return analytical_tip_deflection(
            abs(self.TOTAL_LOAD), self.LENGTH, self.E, I
        )

    def test_hex8_coarse_mesh(self, steel, analytical_deflection):
        """Test with coarse hex8 mesh (5x1x1 elements).

        Coarse meshes underpredict deflection. This test verifies the
        solution is reasonable and converging in the right direction.
        """
        mesh = generate_hex8_cantilever_mesh(
            self.LENGTH, self.WIDTH, self.HEIGHT,
            nx=5, ny=1, nz=1,
        )

        # Build model with distributed load
        base_model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
        )
        model = _apply_distributed_load(mesh, base_model, self.LENGTH, self.TOTAL_LOAD)

        # Solve
        results = solve(model)
        disp = results.displacement()

        # Find tip nodes (at x=LENGTH)
        tip_nodes = Nodes.where(x=self.LENGTH).evaluate(mesh)
        tip_z_displacements = [disp[n, 2] for n in tip_nodes]
        avg_tip_deflection = np.mean(tip_z_displacements)

        # For coarse mesh, expect underprediction
        # Should be within 50% of analytical (this is a coarse mesh)
        assert avg_tip_deflection < 0, "Tip should deflect downward"
        fea_deflection = abs(avg_tip_deflection)

        # Coarse hex8 mesh will be stiffer than reality
        # Accept if within factor of 2 for this coarse discretization
        assert fea_deflection > 0.3 * analytical_deflection, (
            f"FEA deflection {fea_deflection:.6e} too small, "
            f"expected > {0.3 * analytical_deflection:.6e}"
        )
        assert fea_deflection < 2.0 * analytical_deflection, (
            f"FEA deflection {fea_deflection:.6e} too large, "
            f"expected < {2.0 * analytical_deflection:.6e}"
        )

    def test_hex8_medium_mesh(self, steel, analytical_deflection):
        """Test with medium hex8 mesh (10x2x2 elements).

        Medium mesh should be closer to analytical solution.
        """
        mesh = generate_hex8_cantilever_mesh(
            self.LENGTH, self.WIDTH, self.HEIGHT,
            nx=10, ny=2, nz=2,
        )

        base_model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
        )
        model = _apply_distributed_load(mesh, base_model, self.LENGTH, self.TOTAL_LOAD)

        results = solve(model)
        disp = results.displacement()

        tip_nodes = Nodes.where(x=self.LENGTH).evaluate(mesh)
        tip_z_displacements = [disp[n, 2] for n in tip_nodes]
        avg_tip_deflection = abs(np.mean(tip_z_displacements))

        # Medium mesh should be within 50% of analytical
        relative_error = abs(avg_tip_deflection - analytical_deflection) / analytical_deflection
        assert relative_error < 0.50, (
            f"Medium mesh relative error {relative_error:.1%} exceeds 50%. "
            f"FEA: {avg_tip_deflection:.6e}, Analytical: {analytical_deflection:.6e}"
        )

    def test_hex8_fine_mesh(self, steel, analytical_deflection):
        """Test with fine hex8 mesh (20x4x4 elements).

        Fine mesh should converge close to analytical solution.
        """
        mesh = generate_hex8_cantilever_mesh(
            self.LENGTH, self.WIDTH, self.HEIGHT,
            nx=20, ny=4, nz=4,
        )

        base_model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
        )
        model = _apply_distributed_load(mesh, base_model, self.LENGTH, self.TOTAL_LOAD)

        results = solve(model)
        disp = results.displacement()

        tip_nodes = Nodes.where(x=self.LENGTH).evaluate(mesh)
        tip_z_displacements = [disp[n, 2] for n in tip_nodes]
        avg_tip_deflection = abs(np.mean(tip_z_displacements))

        # Fine mesh should be within 30% of analytical
        relative_error = abs(avg_tip_deflection - analytical_deflection) / analytical_deflection
        assert relative_error < 0.30, (
            f"Fine mesh relative error {relative_error:.1%} exceeds 30%. "
            f"FEA: {avg_tip_deflection:.6e}, Analytical: {analytical_deflection:.6e}"
        )

    def test_mesh_convergence(self, steel, analytical_deflection):
        """Test that refinement improves accuracy (convergence study).

        This test verifies that finer meshes produce more accurate results,
        which is a fundamental property of convergent FEA formulations.
        """
        mesh_configs = [
            (5, 1, 1),    # Coarse
            (10, 2, 2),   # Medium
            (20, 4, 4),   # Fine
        ]

        deflections = []
        for nx, ny, nz in mesh_configs:
            mesh = generate_hex8_cantilever_mesh(
                self.LENGTH, self.WIDTH, self.HEIGHT,
                nx=nx, ny=ny, nz=nz,
            )

            base_model = (
                Model(mesh, materials={"steel": steel})
                .assign(Elements.all(), material="steel")
                .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
            )
            model = _apply_distributed_load(mesh, base_model, self.LENGTH, self.TOTAL_LOAD)

            results = solve(model)
            disp = results.displacement()

            tip_nodes = Nodes.where(x=self.LENGTH).evaluate(mesh)
            avg_deflection = abs(np.mean([disp[n, 2] for n in tip_nodes]))
            deflections.append(avg_deflection)

        # Each refinement should get closer to analytical
        errors = [abs(d - analytical_deflection) for d in deflections]

        # Monotonic convergence: each finer mesh should have smaller error
        for i in range(len(errors) - 1):
            assert errors[i+1] <= errors[i] * 1.1, (
                f"Convergence violation: error[{i+1}]={errors[i+1]:.6e} > "
                f"error[{i}]={errors[i]:.6e}"
            )


class TestCantileverBeamFixedNodes:
    """Test that fixed end has zero displacement."""

    def test_fixed_end_zero_displacement(self):
        """Verify nodes at x=0 have zero displacement."""
        mesh = generate_hex8_cantilever_mesh(
            length=10.0, width=1.0, height=1.0,
            nx=5, ny=1, nz=1,
        )

        steel = Material("steel", e=200e9, nu=0.3)
        base_model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
        )
        model = _apply_distributed_load(mesh, base_model, 10.0, -1000)

        results = solve(model)
        disp = results.displacement()

        # All nodes at x=0 should have zero displacement
        fixed_nodes = Nodes.where(x=0).evaluate(mesh)
        for node in fixed_nodes:
            assert np.allclose(disp[node], 0.0, atol=1e-12), (
                f"Node {node} at x=0 has non-zero displacement: {disp[node]}"
            )


class TestCantileverSymmetry:
    """Test symmetry of the cantilever beam solution."""

    def test_symmetric_deflection_about_midplane(self):
        """For symmetric loading, deflection should be symmetric about y=width/2."""
        width = 1.0
        mesh = generate_hex8_cantilever_mesh(
            length=10.0, width=width, height=1.0,
            nx=10, ny=2, nz=2,
        )

        steel = Material("steel", e=200e9, nu=0.3)
        base_model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
        )
        model = _apply_distributed_load(mesh, base_model, 10.0, -1000)

        results = solve(model)
        disp = results.displacement()
        nodes = np.array(mesh.coords)  # Get node coordinates as numpy array

        # Compare displacements at symmetric positions about y=0.5
        tip_nodes = Nodes.where(x=10.0).evaluate(mesh)

        # Group nodes by their z-coordinate
        z_groups = {}
        for n in tip_nodes:
            z_key = round(nodes[n, 2], 6)
            if z_key not in z_groups:
                z_groups[z_key] = []
            z_groups[z_key].append(n)

        # For each z-level, nodes at y and (width-y) should have same z-displacement
        for z_key, node_list in z_groups.items():
            y_disp_pairs = {}
            for n in node_list:
                y = nodes[n, 1]
                y_sym = width - y
                key = min(round(y, 6), round(y_sym, 6))
                if key not in y_disp_pairs:
                    y_disp_pairs[key] = []
                y_disp_pairs[key].append(disp[n, 2])

            for key, disps in y_disp_pairs.items():
                if len(disps) == 2:
                    assert np.isclose(disps[0], disps[1], rtol=1e-6), (
                        f"Asymmetric z-displacement at y={key}: {disps[0]} vs {disps[1]}"
                    )


class TestCantileverLoadScaling:
    """Test that deflection scales linearly with load."""

    def test_linear_load_deflection_relationship(self):
        """Deflection should double when load doubles."""
        mesh = generate_hex8_cantilever_mesh(
            length=10.0, width=1.0, height=1.0,
            nx=10, ny=2, nz=2,
        )
        steel = Material("steel", e=200e9, nu=0.3)

        deflections = []
        loads = [-500, -1000, -2000]  # N (total load)

        for load in loads:
            base_model = (
                Model(mesh, materials={"steel": steel})
                .assign(Elements.all(), material="steel")
                .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
            )
            model = _apply_distributed_load(mesh, base_model, 10.0, load)

            results = solve(model)
            disp = results.displacement()

            tip_nodes = Nodes.where(x=10.0).evaluate(mesh)
            avg_deflection = np.mean([disp[n, 2] for n in tip_nodes])
            deflections.append(avg_deflection)

        # Check linearity: deflection / load should be constant
        ratios = [d / l for d, l in zip(deflections, loads)]
        for i in range(1, len(ratios)):
            assert np.isclose(ratios[i], ratios[0], rtol=1e-6), (
                f"Non-linear load-deflection: ratio[{i}]={ratios[i]:.6e} "
                f"vs ratio[0]={ratios[0]:.6e}"
            )


class TestCantileverMaterialScaling:
    """Test that deflection scales inversely with Young's modulus."""

    def test_inverse_stiffness_relationship(self):
        """Deflection should halve when E doubles."""
        mesh = generate_hex8_cantilever_mesh(
            length=10.0, width=1.0, height=1.0,
            nx=10, ny=2, nz=2,
        )

        E_values = [100e9, 200e9, 400e9]  # Pa
        deflections = []

        for E in E_values:
            material = Material("test", e=E, nu=0.3)
            base_model = (
                Model(mesh, materials={"test": material})
                .assign(Elements.all(), material="test")
                .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
            )
            model = _apply_distributed_load(mesh, base_model, 10.0, -1000)

            results = solve(model)
            disp = results.displacement()

            tip_nodes = Nodes.where(x=10.0).evaluate(mesh)
            avg_deflection = abs(np.mean([disp[n, 2] for n in tip_nodes]))
            deflections.append(avg_deflection)

        # Check inverse relationship: deflection * E should be constant
        products = [d * E for d, E in zip(deflections, E_values)]
        for i in range(1, len(products)):
            assert np.isclose(products[i], products[0], rtol=1e-6), (
                f"Non-inverse E-deflection: product[{i}]={products[i]:.6e} "
                f"vs product[0]={products[0]:.6e}"
            )
