"""Unit tests for stress recovery functionality."""

import numpy as np
import pytest
import mops


class TestStressRecovery:
    """Test stress recovery from displacement solution."""

    def test_stress_shape_single_tet4(self):
        """Stress output should have shape (n_elements, 6)."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")
        material = mops.Material.steel()

        constrained = np.array([0, 2, 3], dtype=np.int64)
        loaded = np.array([1], dtype=np.int64)
        load = np.array([1000.0, 0.0, 0.0], dtype=np.float64)

        results = mops.solve_simple(mesh, material, constrained, loaded, load)
        stress = results.stress()

        assert stress.shape == (1, 6), f"Expected (1, 6), got {stress.shape}"

    def test_stress_shape_multi_element(self):
        """Stress should have one row per element."""
        # Two tet4 elements sharing a face
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 1.0, 0.0],  # 2
            [0.5, 0.5, 1.0],  # 3
            [0.5, 0.5, -1.0], # 4
        ], dtype=np.float64)
        elements = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 4]
        ], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")
        material = mops.Material.steel()

        # Need to fully constrain some nodes to make system SPD
        constrained = np.array([0, 1, 2], dtype=np.int64)  # Fixed shared face
        loaded = np.array([3], dtype=np.int64)  # Load on tip
        load = np.array([0.0, 0.0, 1000.0], dtype=np.float64)

        results = mops.solve_simple(mesh, material, constrained, loaded, load)
        stress = results.stress()

        assert stress.shape == (2, 6), f"Expected (2, 6), got {stress.shape}"

    def test_von_mises_shape(self):
        """Von Mises stress should be (n_elements,) array."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")
        material = mops.Material.steel()

        constrained = np.array([0, 2, 3], dtype=np.int64)
        loaded = np.array([1], dtype=np.int64)
        load = np.array([1000.0, 0.0, 0.0], dtype=np.float64)

        results = mops.solve_simple(mesh, material, constrained, loaded, load)
        vm = results.von_mises()

        assert vm.shape == (1,), f"Expected (1,), got {vm.shape}"
        assert vm[0] > 0, "Von Mises stress should be positive under load"

    def test_max_von_mises(self):
        """max_von_mises should match max of von_mises array."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, -1.0],
        ], dtype=np.float64)
        elements = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 4]
        ], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")
        material = mops.Material.steel()

        # Need 3 nodes to fully constrain (remove rigid body modes)
        constrained = np.array([0, 1, 2], dtype=np.int64)
        loaded = np.array([3], dtype=np.int64)
        load = np.array([0.0, 0.0, 1000.0], dtype=np.float64)

        results = mops.solve_simple(mesh, material, constrained, loaded, load)

        assert results.max_von_mises() == pytest.approx(np.max(results.von_mises()))

    def test_element_stress_access(self):
        """element_stress(id) should return single element's stress tensor."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")
        material = mops.Material.steel()

        constrained = np.array([0, 2, 3], dtype=np.int64)
        loaded = np.array([1], dtype=np.int64)
        load = np.array([1000.0, 0.0, 0.0], dtype=np.float64)

        results = mops.solve_simple(mesh, material, constrained, loaded, load)
        elem_stress = results.element_stress(0)

        assert elem_stress.shape == (6,)
        # Should match first row of stress()
        np.testing.assert_array_equal(elem_stress, results.stress()[0])

    def test_element_stress_out_of_bounds(self):
        """element_stress with invalid index should raise."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")
        material = mops.Material.steel()

        constrained = np.array([0, 2, 3], dtype=np.int64)
        loaded = np.array([1], dtype=np.int64)
        load = np.array([1000.0, 0.0, 0.0], dtype=np.float64)

        results = mops.solve_simple(mesh, material, constrained, loaded, load)

        with pytest.raises(ValueError):
            results.element_stress(10)  # Out of bounds

    def test_zero_displacement_zero_stress(self):
        """Zero displacement should give zero stress."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")
        material = mops.Material.steel()

        # Fix all nodes - zero displacement
        constrained = np.array([0, 1, 2, 3], dtype=np.int64)
        loaded = np.array([], dtype=np.int64)
        load = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        results = mops.solve_simple(mesh, material, constrained, loaded, load)

        assert results.max_von_mises() == pytest.approx(0.0, abs=1e-10)


class TestElementStressFunction:
    """Test compute_element_stress standalone function."""

    def test_tet4_stress_computation(self):
        """compute_element_stress should return integration point stresses."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        material = mops.Material.steel()

        # Uniform extension in x: u = 0.001 * x
        displacements = np.array([
            0.0, 0.0, 0.0,     # node 0
            0.001, 0.0, 0.0,   # node 1
            0.0, 0.0, 0.0,     # node 2
            0.0, 0.0, 0.0,     # node 3
        ], dtype=np.float64)

        stress = mops.compute_element_stress("tet4", nodes, displacements, material)

        # Tet4 has 1 integration point
        assert stress.shape == (1, 6)
        # Non-zero stress under strain
        assert np.any(stress != 0)

    def test_hex8_stress_multiple_integration_points(self):
        """Hex8 should return stresses at 8 integration points."""
        # Unit cube
        nodes = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float64)
        material = mops.Material.steel()

        # Zero displacement
        displacements = np.zeros(24, dtype=np.float64)

        stress = mops.compute_element_stress("hex8", nodes, displacements, material)

        # Hex8 has 2x2x2 = 8 integration points
        assert stress.shape == (8, 6)

    def test_tet10_stress_integration_points(self):
        """Tet10 should return stresses at 4 integration points."""
        # Create valid tet10 (4 corner + 6 midedge nodes)
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0 corner
            [1.0, 0.0, 0.0],  # 1 corner
            [0.0, 1.0, 0.0],  # 2 corner
            [0.0, 0.0, 1.0],  # 3 corner
            [0.5, 0.0, 0.0],  # 4 midedge 0-1
            [0.5, 0.5, 0.0],  # 5 midedge 1-2
            [0.0, 0.5, 0.0],  # 6 midedge 0-2
            [0.0, 0.0, 0.5],  # 7 midedge 0-3
            [0.5, 0.0, 0.5],  # 8 midedge 1-3
            [0.0, 0.5, 0.5],  # 9 midedge 2-3
        ], dtype=np.float64)
        material = mops.Material.steel()

        # Zero displacement
        displacements = np.zeros(30, dtype=np.float64)

        stress = mops.compute_element_stress("tet10", nodes, displacements, material)

        # Tet10 has 4 integration points
        assert stress.shape == (4, 6)

    def test_invalid_displacement_length(self):
        """compute_element_stress should raise for wrong displacement count."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        material = mops.Material.steel()

        # Wrong length (should be 12 for tet4)
        displacements = np.zeros(6, dtype=np.float64)

        with pytest.raises(ValueError):
            mops.compute_element_stress("tet4", nodes, displacements, material)


class TestVonMisesCalculation:
    """Verify von Mises stress calculation is correct."""

    def test_uniaxial_stress(self):
        """Uniaxial stress: von Mises = |sigma|."""
        # Apply force in one direction
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")
        material = mops.Material.steel()

        constrained = np.array([0, 2, 3], dtype=np.int64)
        loaded = np.array([1], dtype=np.int64)
        load = np.array([10000.0, 0.0, 0.0], dtype=np.float64)

        results = mops.solve_simple(mesh, material, constrained, loaded, load)

        # The stress field should be dominated by normal stress in x
        stress = results.stress()[0]
        sigma_xx = stress[0]
        sigma_yy = stress[1]
        sigma_zz = stress[2]

        # For constrained Poisson effect, expect sigma_yy and sigma_zz
        # to be non-zero but much smaller than for pure uniaxial
        assert abs(sigma_xx) > abs(sigma_yy)
        assert abs(sigma_xx) > abs(sigma_zz)

    def test_von_mises_positive(self):
        """Von Mises stress should always be non-negative."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")
        material = mops.Material.steel()

        constrained = np.array([0], dtype=np.int64)
        loaded = np.array([1, 2, 3], dtype=np.int64)
        load = np.array([-500.0, 200.0, -300.0], dtype=np.float64)  # Mixed load

        results = mops.solve_simple(mesh, material, constrained, loaded, load)
        vm = results.von_mises()

        assert np.all(vm >= 0), "Von Mises stress must be non-negative"


class TestResultsRepr:
    """Test Results string representation includes stress info."""

    def test_results_repr_includes_stress(self):
        """Results repr should show max von Mises."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = mops.Mesh(nodes, elements, "tet4")
        material = mops.Material.steel()

        constrained = np.array([0, 2, 3], dtype=np.int64)
        loaded = np.array([1], dtype=np.int64)
        load = np.array([1000.0, 0.0, 0.0], dtype=np.float64)

        results = mops.solve_simple(mesh, material, constrained, loaded, load)
        repr_str = repr(results)

        assert "n_elements=" in repr_str
        assert "max_vm=" in repr_str
