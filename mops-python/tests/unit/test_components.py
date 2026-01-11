"""Tests for component definition and in_component() queries.

Tests cover:
- Model.define_component() for storing named query groups
- Nodes.in_component(), Elements.in_component(), Faces.in_component()
- Component query resolution during evaluation
- Error handling for missing/wrong type components
"""

import numpy as np
import pytest

from mops import Mesh
from mops.model import Model
from mops.query import Nodes, NodeQuery, Elements, ElementQuery, Faces, FaceQuery


class TestComponentDefinition:
    """Tests for Model.define_component()."""

    @pytest.fixture
    def hex8_mesh(self):
        """Create a unit cube hex8 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_define_node_component(self, hex8_mesh):
        """Test defining a node component."""
        model = Model(hex8_mesh)
        model = model.define_component("fixed_nodes", Nodes.where(x=0))

        assert "fixed_nodes" in model._state.components
        assert isinstance(model._state.components["fixed_nodes"], NodeQuery)

    def test_define_element_component(self, hex8_mesh):
        """Test defining an element component."""
        model = Model(hex8_mesh)
        model = model.define_component("all_elements", Elements.all())

        assert "all_elements" in model._state.components
        assert isinstance(model._state.components["all_elements"], ElementQuery)

    def test_define_face_component(self, hex8_mesh):
        """Test defining a face component."""
        model = Model(hex8_mesh)
        model = model.define_component("top_face", Faces.where(z=1))

        assert "top_face" in model._state.components
        assert isinstance(model._state.components["top_face"], FaceQuery)

    def test_define_multiple_components(self, hex8_mesh):
        """Test defining multiple components."""
        model = (
            Model(hex8_mesh)
            .define_component("fixed", Nodes.where(x=0))
            .define_component("loaded", Nodes.where(x=1))
            .define_component("all_elem", Elements.all())
        )

        assert len(model._state.components) == 3
        assert "fixed" in model._state.components
        assert "loaded" in model._state.components
        assert "all_elem" in model._state.components

    def test_define_component_immutability(self, hex8_mesh):
        """Test that define_component returns a new Model."""
        model1 = Model(hex8_mesh)
        model2 = model1.define_component("test", Nodes.all())

        assert model1 is not model2
        assert "test" not in model1._state.components
        assert "test" in model2._state.components


class TestNodesInComponent:
    """Tests for Nodes.in_component() query."""

    @pytest.fixture
    def hex8_mesh(self):
        """Create a unit cube hex8 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_in_component_returns_nodequery(self):
        """Test that Nodes.in_component returns a NodeQuery."""
        query = Nodes.in_component("test_component")

        assert isinstance(query, NodeQuery)
        assert query.component_name == "test_component"

    def test_in_component_evaluate_with_components(self, hex8_mesh):
        """Test evaluating a component query with components dict."""
        # Define component as nodes at x=0
        components = {"fixed_nodes": Nodes.where(x=0)}

        query = Nodes.in_component("fixed_nodes")
        result = query.evaluate(hex8_mesh._core, components)

        # Should get nodes 0, 3, 4, 7 (all with x=0)
        expected = {0, 3, 4, 7}
        assert set(result) == expected

    def test_in_component_evaluate_without_components_raises(self, hex8_mesh):
        """Test that evaluating without components dict raises error."""
        query = Nodes.in_component("fixed_nodes")

        with pytest.raises(ValueError, match="requires components dict"):
            query.evaluate(hex8_mesh._core)

    def test_in_component_missing_component_raises(self, hex8_mesh):
        """Test that referencing undefined component raises error."""
        components = {"other": Nodes.where(x=0)}
        query = Nodes.in_component("nonexistent")

        with pytest.raises(ValueError, match="Unknown component"):
            query.evaluate(hex8_mesh._core, components)

    def test_in_component_wrong_type_raises(self, hex8_mesh):
        """Test that using wrong component type raises error."""
        # Define an element component, try to use with Nodes.in_component
        components = {"elements": Elements.all()}
        query = Nodes.in_component("elements")

        with pytest.raises(TypeError, match="not a node query"):
            query.evaluate(hex8_mesh._core, components)

    def test_in_component_repr(self):
        """Test repr for Nodes.in_component query."""
        query = Nodes.in_component("my_component")
        repr_str = repr(query)

        assert "Nodes.in_component" in repr_str
        assert "my_component" in repr_str


class TestElementsInComponent:
    """Tests for Elements.in_component() query."""

    @pytest.fixture
    def multi_hex_mesh(self):
        """Create two hex8 elements stacked in z direction."""
        nodes = np.array([
            # Bottom cube nodes
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
            # Top cube nodes
            [0.0, 0.0, 2.0],  # 8
            [1.0, 0.0, 2.0],  # 9
            [1.0, 1.0, 2.0],  # 10
            [0.0, 1.0, 2.0],  # 11
        ], dtype=np.float64)
        elements = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],    # Element 0 (bottom cube)
            [4, 5, 6, 7, 8, 9, 10, 11],  # Element 1 (top cube)
        ], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_in_component_returns_elementquery(self):
        """Test that Elements.in_component returns an ElementQuery."""
        query = Elements.in_component("bracket")

        assert isinstance(query, ElementQuery)
        assert query.component_name == "bracket"

    def test_in_component_evaluate_with_components(self, multi_hex_mesh):
        """Test evaluating an element component query."""
        components = {"bottom_element": Elements.by_indices([0])}

        query = Elements.in_component("bottom_element")
        result = query.evaluate(multi_hex_mesh._core, components)

        assert list(result) == [0]

    def test_in_component_evaluate_all_elements(self, multi_hex_mesh):
        """Test evaluating a component that selects all elements."""
        components = {"all": Elements.all()}

        query = Elements.in_component("all")
        result = query.evaluate(multi_hex_mesh._core, components)

        assert set(result) == {0, 1}

    def test_in_component_wrong_type_raises(self, multi_hex_mesh):
        """Test that using wrong component type raises error."""
        # Define a node component, try to use with Elements.in_component
        components = {"nodes": Nodes.where(x=0)}
        query = Elements.in_component("nodes")

        with pytest.raises(TypeError, match="not an element query"):
            query.evaluate(multi_hex_mesh._core, components)

    def test_in_component_repr(self):
        """Test repr for Elements.in_component query."""
        query = Elements.in_component("bracket")
        repr_str = repr(query)

        assert "Elements.in_component" in repr_str
        assert "bracket" in repr_str


class TestFacesInComponent:
    """Tests for Faces.in_component() query."""

    @pytest.fixture
    def hex8_mesh(self):
        """Create a unit cube hex8 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_in_component_returns_facequery(self):
        """Test that Faces.in_component returns a FaceQuery."""
        query = Faces.in_component("load_surface")

        assert isinstance(query, FaceQuery)
        assert query.component_name == "load_surface"

    def test_in_component_evaluate_with_components(self, hex8_mesh):
        """Test evaluating a face component query."""
        # Define component as top face (z=1)
        components = {"top_face": Faces.where(z=1.0)}

        query = Faces.in_component("top_face")
        result = query.evaluate(hex8_mesh, components)

        # Should get one face (top face)
        assert result.shape[0] == 1
        assert result[0, 1] == 1  # top face local index

    def test_in_component_boundary_faces(self, hex8_mesh):
        """Test component with boundary face query."""
        components = {"boundary": Faces.on_boundary()}

        query = Faces.in_component("boundary")
        result = query.evaluate(hex8_mesh, components)

        # All 6 faces are boundary
        assert result.shape[0] == 6

    def test_in_component_wrong_type_raises(self, hex8_mesh):
        """Test that using wrong component type raises error."""
        # Define a node component, try to use with Faces.in_component
        components = {"nodes": Nodes.where(x=0)}
        query = Faces.in_component("nodes")

        with pytest.raises(TypeError, match="not a face query"):
            query.evaluate(hex8_mesh, components)

    def test_in_component_repr(self):
        """Test repr for Faces.in_component query."""
        query = Faces.in_component("load_surface")
        repr_str = repr(query)

        assert "Faces.in_component" in repr_str
        assert "load_surface" in repr_str


class TestNestedComponentQueries:
    """Tests for components that reference other components."""

    @pytest.fixture
    def hex8_mesh(self):
        """Create a unit cube hex8 element mesh."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return Mesh.from_arrays(nodes, elements, "hex8")

    def test_union_with_component(self, hex8_mesh):
        """Test union of regular query with component query."""
        components = {
            "left_nodes": Nodes.where(x=0),
        }

        # Union of right nodes (x=1) with left nodes component
        combined = Nodes.where(x=1).union(Nodes.in_component("left_nodes"))
        result = combined.evaluate(hex8_mesh._core, components)

        # Should get nodes at x=0 and x=1
        expected = {0, 1, 2, 3, 4, 5, 6, 7}  # All nodes
        assert set(result) == expected

    def test_subtract_with_component(self, hex8_mesh):
        """Test subtraction using component query."""
        components = {
            "left_nodes": Nodes.where(x=0),
        }

        # All nodes minus left nodes
        remaining = Nodes.all().subtract(Nodes.in_component("left_nodes"))
        result = remaining.evaluate(hex8_mesh._core, components)

        # Should get nodes at x=1 (indices 1, 2, 5, 6)
        expected = {1, 2, 5, 6}
        assert set(result) == expected


class TestComponentQueryDocstrings:
    """Tests to verify docstrings exist and are correct."""

    def test_nodes_in_component_has_docstring(self):
        """Test that Nodes.in_component has a docstring."""
        assert Nodes.in_component.__doc__ is not None
        assert "component" in Nodes.in_component.__doc__.lower()
        assert "APDL" in Nodes.in_component.__doc__ or "CMSEL" in Nodes.in_component.__doc__

    def test_elements_in_component_has_docstring(self):
        """Test that Elements.in_component has a docstring."""
        assert Elements.in_component.__doc__ is not None
        assert "component" in Elements.in_component.__doc__.lower()

    def test_faces_in_component_has_docstring(self):
        """Test that Faces.in_component has a docstring."""
        assert Faces.in_component.__doc__ is not None
        assert "component" in Faces.in_component.__doc__.lower()
