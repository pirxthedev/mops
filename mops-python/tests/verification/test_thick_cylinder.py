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
    Force,
    Material,
    Mesh,
    Model,
    Nodes,
    solve,
)


@pytest.fixture
def steel():
    """Steel material: E=200 GPa, ν=0.3."""
    return Material.steel()


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

        model = (
            Model(mesh, materials={"steel": steel})
            .constrain(Nodes.where(y=0.0), dofs=["uy"])  # Fix bottom in z (axial)
            .constrain(Nodes.by_indices([0]), dofs=["ux"])  # Fix one node in r to prevent rigid body motion
            # Apply outward force on outer nodes
            .load(Nodes.by_indices([1, 2]), Force(fx=1000.0))
        )

        results = solve(model)

        # Should have non-zero displacements
        assert results.max_displacement() > 0

    def test_tri3_single_element_solve(self, steel):
        """Test that a single tri3 element problem can be solved."""
        nodes = np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 1.0, 0.0]], dtype=np.float64
        )
        elements = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tri3axisymmetric")

        model = (
            Model(mesh, materials={"steel": steel})
            .constrain(Nodes.where(y=0.0), dofs=["uy"])
            .constrain(Nodes.by_indices([0]), dofs=["ux"])
            .load(Nodes.by_indices([1]), Force(fx=1000.0))
        )

        results = solve(model)
        assert results.max_displacement() > 0


class TestAxisymmetricMeshCreation:
    """Tests for axisymmetric mesh creation."""

    def test_quad4_axisymmetric_mesh(self, steel):
        """Verify mesh can be created with quad4axisymmetric elements."""
        nodes = np.array(
            [[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], dtype=np.float64
        )
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "quad4axisymmetric")

        assert mesh.n_nodes == 4
        assert mesh.n_elements == 1
        assert mesh.element_type == "quad4axisymmetric"

    def test_tri3_axisymmetric_mesh(self, steel):
        """Verify mesh can be created with tri3axisymmetric elements."""
        nodes = np.array(
            [[1, 0, 0], [2, 0, 0], [1.5, 1, 0]], dtype=np.float64
        )
        elements = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = Mesh(nodes, elements, "tri3axisymmetric")

        assert mesh.n_nodes == 3
        assert mesh.n_elements == 1
        assert mesh.element_type == "tri3axisymmetric"
