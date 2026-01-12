"""Verification tests for axisymmetric elements: thick-walled cylinder under internal pressure.

This module tests the axisymmetric element implementations (Tri3Axisymmetric, Quad4Axisymmetric)
against the classical Lamé solution for a thick-walled cylinder under internal pressure.

Analytical Solution (Lamé):
--------------------------
For a thick-walled cylinder with:
- Inner radius: a
- Outer radius: b
- Internal pressure: p
- Material: E (Young's modulus), ν (Poisson's ratio)

Radial stress:
    σ_r(r) = p * a² / (b² - a²) * (1 - b²/r²)

Hoop (circumferential) stress:
    σ_θ(r) = p * a² / (b² - a²) * (1 + b²/r²)

Radial displacement:
    u_r(r) = p * a² / (E * (b² - a²)) * ((1-ν)*r + (1+ν)*b²/r)

Key values:
- At inner surface (r=a):
    σ_r(a) = -p (compressive, equals applied pressure)
    σ_θ(a) = p * (a² + b²) / (b² - a²) (maximum hoop stress)

- At outer surface (r=b):
    σ_r(b) = 0 (traction-free)
    σ_θ(b) = 2 * p * a² / (b² - a²)

References:
- Timoshenko & Goodier, "Theory of Elasticity" (3rd ed., 1970), Section 4.2
- Boresi & Schmidt, "Advanced Mechanics of Materials" (6th ed.)
- Young & Budynas, "Roark's Formulas for Stress and Strain" (8th ed.)

Coordinate System:
-----------------
Axisymmetric elements use the r-z plane where:
- r = radial coordinate (x-coordinate in Point3)
- z = axial coordinate (y-coordinate in Point3)
- Axis of revolution: z-axis (r=0)

The third coordinate (z in Point3) is ignored for axisymmetric elements.
"""

import numpy as np
import pytest

from mops import (
    Elements,
    Material,
    Mesh,
    Model,
    Nodes,
    Pressure,
    solve,
)


class LameSolution:
    """Analytical Lamé solution for thick-walled cylinder under internal pressure."""

    def __init__(self, a: float, b: float, p: float, E: float, nu: float):
        """Initialize Lamé solution parameters.

        Args:
            a: Inner radius
            b: Outer radius
            p: Internal pressure (positive = outward)
            E: Young's modulus
            nu: Poisson's ratio
        """
        self.a = a
        self.b = b
        self.p = p
        self.E = E
        self.nu = nu
        self.k = a**2 / (b**2 - a**2)  # Common factor

    def sigma_r(self, r: float) -> float:
        """Radial stress at radius r."""
        return self.p * self.k * (1 - self.b**2 / r**2)

    def sigma_theta(self, r: float) -> float:
        """Hoop (circumferential) stress at radius r."""
        return self.p * self.k * (1 + self.b**2 / r**2)

    def u_r(self, r: float) -> float:
        """Radial displacement at radius r."""
        factor = self.p * self.a**2 / (self.E * (self.b**2 - self.a**2))
        return factor * ((1 - self.nu) * r + (1 + self.nu) * self.b**2 / r)

    @property
    def sigma_r_inner(self) -> float:
        """Radial stress at inner surface (equals -p)."""
        return self.sigma_r(self.a)

    @property
    def sigma_theta_inner(self) -> float:
        """Hoop stress at inner surface (maximum)."""
        return self.sigma_theta(self.a)

    @property
    def sigma_r_outer(self) -> float:
        """Radial stress at outer surface (should be ~0)."""
        return self.sigma_r(self.b)

    @property
    def sigma_theta_outer(self) -> float:
        """Hoop stress at outer surface."""
        return self.sigma_theta(self.b)

    @property
    def u_r_inner(self) -> float:
        """Radial displacement at inner surface."""
        return self.u_r(self.a)

    @property
    def u_r_outer(self) -> float:
        """Radial displacement at outer surface."""
        return self.u_r(self.b)


def generate_quad4_cylinder_mesh(
    a: float,
    b: float,
    h: float,
    nr: int,
    nz: int,
    element_type: str = "quad4axisymmetric",
) -> Mesh:
    """Generate a structured quad4 axisymmetric mesh for a cylinder cross-section.

    The mesh covers a rectangular region in the r-z plane:
    - r ranges from a (inner) to b (outer)
    - z ranges from 0 to h (height)

    Args:
        a: Inner radius
        b: Outer radius
        h: Cylinder height (axial extent)
        nr: Number of elements in radial direction
        nz: Number of elements in axial direction
        element_type: Element type string

    Returns:
        Mesh: A quad4axisymmetric mesh
    """
    # Generate node coordinates in r-z plane
    # Note: For axisymmetric, we store (r, z, 0) in Point3
    r_coords = np.linspace(a, b, nr + 1)
    z_coords = np.linspace(0, h, nz + 1)

    nodes = []
    for j, z in enumerate(z_coords):
        for i, r in enumerate(r_coords):
            nodes.append([r, z, 0.0])
    nodes = np.array(nodes, dtype=np.float64)

    def node_index(i, j):
        return i + j * (nr + 1)

    elements = []
    for j in range(nz):
        for i in range(nr):
            # Quad4 node ordering (counterclockwise)
            n0 = node_index(i, j)
            n1 = node_index(i + 1, j)
            n2 = node_index(i + 1, j + 1)
            n3 = node_index(i, j + 1)
            elements.append([n0, n1, n2, n3])

    elements = np.array(elements, dtype=np.int64)
    return Mesh(nodes, elements, element_type)


def generate_tri3_cylinder_mesh(
    a: float,
    b: float,
    h: float,
    nr: int,
    nz: int,
    element_type: str = "tri3axisymmetric",
) -> Mesh:
    """Generate a structured tri3 axisymmetric mesh for a cylinder cross-section.

    Each quad cell is divided into 2 triangles (diagonal split).

    Args:
        a: Inner radius
        b: Outer radius
        h: Cylinder height (axial extent)
        nr: Number of cells in radial direction (each split into 2 triangles)
        nz: Number of cells in axial direction

    Returns:
        Mesh: A tri3axisymmetric mesh
    """
    r_coords = np.linspace(a, b, nr + 1)
    z_coords = np.linspace(0, h, nz + 1)

    nodes = []
    for j, z in enumerate(z_coords):
        for i, r in enumerate(r_coords):
            nodes.append([r, z, 0.0])
    nodes = np.array(nodes, dtype=np.float64)

    def node_index(i, j):
        return i + j * (nr + 1)

    elements = []
    for j in range(nz):
        for i in range(nr):
            n0 = node_index(i, j)
            n1 = node_index(i + 1, j)
            n2 = node_index(i + 1, j + 1)
            n3 = node_index(i, j + 1)
            # Split into 2 triangles with diagonal from n0 to n2
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

    elements = np.array(elements, dtype=np.int64)
    return Mesh(nodes, elements, element_type)


# Problem parameters
INNER_RADIUS = 0.1  # 100 mm
OUTER_RADIUS = 0.2  # 200 mm
PRESSURE = 10e6  # 10 MPa internal pressure
HEIGHT = 0.05  # 50 mm cylinder section (doesn't affect plane strain results)


@pytest.fixture
def steel():
    """Steel material: E=200 GPa, ν=0.3."""
    return Material.steel()


@pytest.fixture
def lame_solution(steel):
    """Analytical Lamé solution for the test problem."""
    return LameSolution(
        a=INNER_RADIUS,
        b=OUTER_RADIUS,
        p=PRESSURE,
        E=steel.e,
        nu=steel.nu,
    )


class TestQuad4Axisymmetric:
    """Tests for Quad4Axisymmetric element with thick-walled cylinder."""

    def test_mesh_creation(self, steel):
        """Verify mesh can be created with quad4axisymmetric elements."""
        mesh = generate_quad4_cylinder_mesh(
            a=INNER_RADIUS, b=OUTER_RADIUS, h=HEIGHT, nr=4, nz=2
        )

        assert mesh.n_nodes == 5 * 3  # (4+1) * (2+1)
        assert mesh.n_elements == 4 * 2
        assert mesh.element_type == "quad4axisymmetric"

    def test_radial_displacement_coarse(self, steel, lame_solution):
        """Test radial displacement with a coarse mesh."""
        mesh = generate_quad4_cylinder_mesh(
            a=INNER_RADIUS, b=OUTER_RADIUS, h=HEIGHT, nr=4, nz=2
        )

        model = (
            Model(mesh, steel)
            # Fix bottom face in z-direction (axial constraint)
            # For axisymmetric: y is z-axis (axial), x is r-axis (radial)
            .constrain(Nodes.where(y=0.0), dofs=["uy"])
            # Apply internal pressure on inner surface
            .load(Pressure(PRESSURE), Elements.where(centroid_x__lt=(INNER_RADIUS + OUTER_RADIUS) / 2).faces().where(normal_x=-1))
        )

        results = solve(model)

        # Check radial displacement at inner surface
        inner_nodes = results.nodes.where(x=INNER_RADIUS)
        u_r_fem = np.abs(inner_nodes.displacement[:, 0]).mean()
        u_r_analytical = lame_solution.u_r_inner

        # Coarse mesh, expect ~10% accuracy
        assert abs(u_r_fem - u_r_analytical) / u_r_analytical < 0.15

    def test_hoop_stress_convergence(self, steel, lame_solution):
        """Test that hoop stress converges to analytical solution with mesh refinement."""
        errors = []
        mesh_sizes = [2, 4, 8, 16]

        for nr in mesh_sizes:
            mesh = generate_quad4_cylinder_mesh(
                a=INNER_RADIUS, b=OUTER_RADIUS, h=HEIGHT, nr=nr, nz=2
            )

            model = (
                Model(mesh, steel)
                .constrain(Nodes.where(y=0.0), dofs=["uy"])
                # Apply internal pressure on inner surface faces
                # For axisymmetric: faces at r=a (x=INNER_RADIUS) with inward normal
                .load(Pressure(PRESSURE), Elements.where(centroid_x__lt=(INNER_RADIUS + OUTER_RADIUS) / 2).faces().where(normal_x=-1))
            )

            results = solve(model)

            # Get hoop stress (σ_zz in the stress tensor = σ_θθ for axisymmetric)
            # Average stress in elements near inner surface
            inner_elements = results.elements.where(centroid_x__lt=INNER_RADIUS * 1.1)
            # In axisymmetric elements, stress[2] = σ_θθ (hoop stress)
            sigma_theta_fem = np.mean([results.stress(e)[2] for e in inner_elements.indices])
            sigma_theta_analytical = lame_solution.sigma_theta_inner

            error = abs(sigma_theta_fem - sigma_theta_analytical) / sigma_theta_analytical
            errors.append(error)

        # Verify convergence (error should decrease with refinement)
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], f"Error should decrease: {errors}"

        # Fine mesh should be within 5%
        assert errors[-1] < 0.05


class TestTri3Axisymmetric:
    """Tests for Tri3Axisymmetric element with thick-walled cylinder."""

    def test_mesh_creation(self, steel):
        """Verify mesh can be created with tri3axisymmetric elements."""
        mesh = generate_tri3_cylinder_mesh(
            a=INNER_RADIUS, b=OUTER_RADIUS, h=HEIGHT, nr=4, nz=2
        )

        assert mesh.n_nodes == 5 * 3  # (4+1) * (2+1)
        assert mesh.n_elements == 4 * 2 * 2  # Each quad split into 2 triangles
        assert mesh.element_type == "tri3axisymmetric"

    def test_radial_displacement_coarse(self, steel, lame_solution):
        """Test radial displacement with a coarse mesh."""
        mesh = generate_tri3_cylinder_mesh(
            a=INNER_RADIUS, b=OUTER_RADIUS, h=HEIGHT, nr=4, nz=2
        )

        model = (
            Model(mesh, steel)
            .constrain(Nodes.where(y=0.0), dofs=["uy"])
            .load(Pressure(PRESSURE), Elements.where(centroid_x__lt=(INNER_RADIUS + OUTER_RADIUS) / 2).faces().where(normal_x=-1))
        )

        results = solve(model)

        # Check radial displacement at inner surface
        inner_nodes = results.nodes.where(x=INNER_RADIUS)
        u_r_fem = np.abs(inner_nodes.displacement[:, 0]).mean()
        u_r_analytical = lame_solution.u_r_inner

        # Tri3 is lower order, expect ~15-20% accuracy with coarse mesh
        assert abs(u_r_fem - u_r_analytical) / u_r_analytical < 0.25


class TestAxisymmetricElementStiffness:
    """Unit tests for axisymmetric element stiffness matrices."""

    def test_quad4_axisymmetric_stiffness_positive_definite(self, steel):
        """Verify stiffness matrix is positive semi-definite."""
        from mops._core import element_stiffness, Material as CoreMaterial

        # Simple quad in r-z plane
        nodes = np.array(
            [
                [1.0, 0.0, 0.0],  # (r=1, z=0)
                [2.0, 0.0, 0.0],  # (r=2, z=0)
                [2.0, 1.0, 0.0],  # (r=2, z=1)
                [1.0, 1.0, 0.0],  # (r=1, z=1)
            ],
            dtype=np.float64,
        )

        mat = CoreMaterial("test", 200e9, 0.3)
        K = element_stiffness("quad4axisymmetric", nodes, mat)

        # Check symmetry
        assert np.allclose(K, K.T, rtol=1e-10)

        # Check positive semi-definiteness (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(K)
        # Allow small numerical noise for zero eigenvalues (rigid body modes)
        assert np.all(eigenvalues > -1e-6 * np.max(eigenvalues))

    def test_tri3_axisymmetric_stiffness_positive_definite(self, steel):
        """Verify stiffness matrix is positive semi-definite."""
        from mops._core import element_stiffness, Material as CoreMaterial

        # Simple triangle in r-z plane
        nodes = np.array(
            [
                [1.0, 0.0, 0.0],  # (r=1, z=0)
                [2.0, 0.0, 0.0],  # (r=2, z=0)
                [1.5, 1.0, 0.0],  # (r=1.5, z=1)
            ],
            dtype=np.float64,
        )

        mat = CoreMaterial("test", 200e9, 0.3)
        K = element_stiffness("tri3axisymmetric", nodes, mat)

        # Check symmetry
        assert np.allclose(K, K.T, rtol=1e-10)

        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > -1e-6 * np.max(eigenvalues))

    def test_stiffness_matrix_dimensions(self, steel):
        """Verify stiffness matrix has correct dimensions."""
        from mops._core import element_stiffness, Material as CoreMaterial

        mat = CoreMaterial("test", 200e9, 0.3)

        # Quad4: 4 nodes * 2 DOFs = 8x8
        quad_nodes = np.array(
            [[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], dtype=np.float64
        )
        K_quad = element_stiffness("quad4axisymmetric", quad_nodes, mat)
        assert K_quad.shape == (8, 8)

        # Tri3: 3 nodes * 2 DOFs = 6x6
        tri_nodes = np.array([[1, 0, 0], [2, 0, 0], [1.5, 1, 0]], dtype=np.float64)
        K_tri = element_stiffness("tri3axisymmetric", tri_nodes, mat)
        assert K_tri.shape == (6, 6)


class TestAxisymmetricBasicSolve:
    """Basic solve tests for axisymmetric elements."""

    def test_quad4_single_element_solve(self, steel):
        """Test that a single element problem can be solved."""
        # Single quad4 element
        nodes = np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "quad4axisymmetric")

        from mops import Force
        model = (
            Model(mesh, materials={"steel": steel})
            .constrain(Nodes.where(y=0.0), dofs=["uy"])  # Fix bottom in z (axial)
            .constrain(Nodes.by_indices([0]), dofs=["ux"])  # Fix one node in r to prevent rigid body motion
            # Apply outward force on outer nodes
            .load(Nodes.by_indices([1, 2]), Force(fx=1000.0))
        )

        results = solve(model)

        # Should have non-zero displacements
        assert results.max_displacement > 0

    def test_tri3_single_element_solve(self, steel):
        """Test that a single tri3 element problem can be solved."""
        nodes = np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 1.0, 0.0]], dtype=np.float64
        )
        elements = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tri3axisymmetric")

        from mops import Force
        model = (
            Model(mesh, materials={"steel": steel})
            .constrain(Nodes.where(y=0.0), dofs=["uy"])
            .constrain(Nodes.by_indices([0]), dofs=["ux"])
            .load(Nodes.by_indices([1]), Force(fx=1000.0))
        )

        results = solve(model)
        assert results.max_displacement > 0
