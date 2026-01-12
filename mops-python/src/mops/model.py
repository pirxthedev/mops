"""Model class with copy-on-write semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mops._core import Material, Mesh, Results, SolverConfig
    from mops.loads import Load
    from mops.query import ElementQuery, FaceQuery, NodeQuery, Query


@dataclass(frozen=True)
class ModelState:
    """Internal immutable state for Model."""

    mesh: "Mesh"
    materials: dict[str, "Material"] = field(default_factory=dict)
    material_assignments: dict[str, str] = field(default_factory=dict)  # query_repr -> material name (legacy)
    thickness_assignments: dict[str, float] = field(default_factory=dict)  # query_repr -> thickness
    constraints: list[tuple["NodeQuery", list[str], float]] = field(default_factory=list)
    loads: list[tuple["NodeQuery | FaceQuery", "Load"]] = field(default_factory=list)
    components: dict[str, "Query"] = field(default_factory=dict)
    # Store actual query objects for per-element material assignment
    material_queries: dict[str, "ElementQuery"] = field(default_factory=dict)  # query_repr -> ElementQuery


class Model:
    """Immutable FEA model with copy-on-write updates.

    All mutation methods return a new Model instance, leaving the original unchanged.
    This enables safe method chaining and functional programming patterns.

    Example::

        model = (
            Model(mesh, materials={"steel": steel})
            .assign(Elements.all(), material="steel")
            .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
            .load(Nodes.where(x=100), Force(fy=-1000))
        )
    """

    def __init__(
        self,
        mesh: "Mesh",
        materials: dict[str, "Material"] | None = None,
        *,
        _state: ModelState | None = None,
    ):
        """Create a new model from mesh.

        Args:
            mesh: The finite element mesh
            materials: Dictionary of material name -> Material
        """
        if _state is not None:
            self._state = _state
        else:
            self._state = ModelState(
                mesh=mesh,
                materials=materials or {},
            )

    @property
    def mesh(self) -> "Mesh":
        """The finite element mesh."""
        return self._state.mesh

    @property
    def materials(self) -> dict[str, "Material"]:
        """Material definitions."""
        return self._state.materials

    @property
    def thickness(self) -> float | None:
        """Element thickness for 2D plane stress elements.

        Returns the first assigned thickness, or None if no thickness was specified.
        """
        if self._state.thickness_assignments:
            return next(iter(self._state.thickness_assignments.values()))
        return None

    def with_material(self, name: str, material: "Material") -> Model:
        """Return new Model with additional material definition.

        Args:
            name: Name to assign to the material
            material: Material properties

        Returns:
            New Model with the material added
        """
        new_materials = {**self._state.materials, name: material}
        return Model(
            mesh=self._state.mesh,
            _state=ModelState(
                mesh=self._state.mesh,
                materials=new_materials,
                material_assignments=self._state.material_assignments,
                thickness_assignments=self._state.thickness_assignments,
                constraints=self._state.constraints,
                loads=self._state.loads,
                components=self._state.components,
                material_queries=self._state.material_queries,
            ),
        )

    def assign(
        self,
        query: "ElementQuery",
        *,
        material: str,
        thickness: float | None = None,
    ) -> Model:
        """Return new Model with material assigned to selected elements.

        Args:
            query: Element selection query
            material: Name of material to assign (must be in materials dict)
            thickness: Element thickness for 2D plane stress elements.
                       Required for 2D elements, ignored for 3D elements.
                       Default: 1.0 if not specified for 2D elements.

        Returns:
            New Model with material assignment
        """
        if material not in self._state.materials:
            raise ValueError(f"Unknown material: {material}. Available: {list(self._state.materials.keys())}")

        if thickness is not None and thickness <= 0:
            raise ValueError(f"thickness must be positive, got {thickness}")

        # Use query repr as key for backward compatibility
        query_key = repr(query)
        new_assignments = {**self._state.material_assignments, query_key: material}

        # Store the actual query object for evaluation during solve
        new_queries = {**self._state.material_queries, query_key: query}

        # Update thickness assignments if specified
        new_thickness = {**self._state.thickness_assignments}
        if thickness is not None:
            new_thickness[query_key] = thickness

        return Model(
            mesh=self._state.mesh,
            _state=ModelState(
                mesh=self._state.mesh,
                materials=self._state.materials,
                material_assignments=new_assignments,
                thickness_assignments=new_thickness,
                constraints=self._state.constraints,
                loads=self._state.loads,
                components=self._state.components,
                material_queries=new_queries,
            ),
        )

    def constrain(
        self,
        query: "NodeQuery",
        dofs: list[str],
        value: float = 0.0,
    ) -> Model:
        """Return new Model with displacement constraint.

        Args:
            query: Node selection query
            dofs: List of DOFs to constrain ("ux", "uy", "uz", "rx", "ry", "rz")
            value: Prescribed displacement value (default 0.0)

        Returns:
            New Model with constraint added
        """
        valid_dofs = {"ux", "uy", "uz", "rx", "ry", "rz"}
        for dof in dofs:
            if dof not in valid_dofs:
                raise ValueError(f"Invalid DOF: {dof}. Valid: {valid_dofs}")

        new_constraints = [*self._state.constraints, (query, dofs, value)]

        return Model(
            mesh=self._state.mesh,
            _state=ModelState(
                mesh=self._state.mesh,
                materials=self._state.materials,
                material_assignments=self._state.material_assignments,
                thickness_assignments=self._state.thickness_assignments,
                constraints=new_constraints,
                loads=self._state.loads,
                components=self._state.components,
                material_queries=self._state.material_queries,
            ),
        )

    def load(self, query: "NodeQuery | FaceQuery", load: "Load") -> Model:
        """Return new Model with applied load.

        Args:
            query: Node or face selection query
            load: Load to apply (Force, Pressure, Moment)

        Returns:
            New Model with load added
        """
        new_loads = [*self._state.loads, (query, load)]

        return Model(
            mesh=self._state.mesh,
            _state=ModelState(
                mesh=self._state.mesh,
                materials=self._state.materials,
                material_assignments=self._state.material_assignments,
                thickness_assignments=self._state.thickness_assignments,
                constraints=self._state.constraints,
                loads=new_loads,
                components=self._state.components,
                material_queries=self._state.material_queries,
            ),
        )

    def define_component(self, name: str, query: "Query") -> Model:
        """Return new Model with named component group.

        Args:
            name: Name for the component
            query: Selection query defining the component

        Returns:
            New Model with component defined
        """
        new_components = {**self._state.components, name: query}

        return Model(
            mesh=self._state.mesh,
            _state=ModelState(
                mesh=self._state.mesh,
                materials=self._state.materials,
                material_assignments=self._state.material_assignments,
                thickness_assignments=self._state.thickness_assignments,
                constraints=self._state.constraints,
                loads=self._state.loads,
                components=new_components,
                material_queries=self._state.material_queries,
            ),
        )

    def _solve(self, config: "SolverConfig | None" = None) -> "Results":
        """Internal solve method.

        This evaluates all queries and calls the Rust solver.

        Supports both single-material and multi-material analysis:
        - If all elements use the same material, uses the optimized single-material path
        - If different materials are assigned to different element groups, uses
          the multi-material solver with per-element material lookup
        """
        from mops._core import solve_with_forces, solve_with_materials
        from mops.loads import Force, Pressure
        from mops.query import ElementQuery, FaceQuery, NodeQuery

        # Validate model completeness
        if not self._state.materials:
            raise ValueError("No materials defined")
        if not self._state.constraints:
            raise ValueError("No constraints defined - model is unconstrained")
        if not self._state.loads:
            raise ValueError("No loads defined")

        # Determine if we need multi-material solve
        # Build element-to-material mapping by evaluating material_assignments queries
        components = self._state.components
        material_names = list(self._state.materials.keys())
        material_name_to_idx = {name: i for i, name in enumerate(material_names)}

        # Build element -> material index mapping by evaluating stored queries
        element_material_map: dict[int, int] = {}
        mesh = self._state.mesh
        core_mesh = mesh._inner if hasattr(mesh, "_inner") else mesh

        for query_repr, material_name in self._state.material_assignments.items():
            mat_idx = material_name_to_idx[material_name]

            # Get the actual query object from material_queries
            if query_repr in self._state.material_queries:
                query = self._state.material_queries[query_repr]
                # Evaluate the query to get element indices
                element_indices = query.evaluate(core_mesh, components)
                for elem_idx in element_indices:
                    element_material_map[int(elem_idx)] = mat_idx
            else:
                # Fallback: assign to all elements (for backward compatibility)
                for elem_idx in range(core_mesh.n_elements):
                    element_material_map[elem_idx] = mat_idx

        # Check if all elements use the same material
        unique_materials = set(element_material_map.values()) if element_material_map else {0}
        use_multi_material = len(unique_materials) > 1

        # Get the first material as default (for single-material fallback)
        default_material = next(iter(self._state.materials.values()))

        # Get components dict for resolving component queries
        components = self._state.components

        # DOF name to index mapping
        dof_map = {"ux": 0, "uy": 1, "uz": 2}

        # Build constraint array: each row is (node_index, dof_index, value)
        constraint_rows = []
        for query, dofs, value in self._state.constraints:
            # Evaluate the query to get node indices
            node_indices = query.evaluate(self._state.mesh, components)
            for node_idx in node_indices:
                for dof_name in dofs:
                    if dof_name in dof_map:
                        dof_idx = dof_map[dof_name]
                        constraint_rows.append([float(node_idx), float(dof_idx), value])

        # Convert to Nx3 array
        if constraint_rows:
            constraints = np.array(constraint_rows, dtype=np.float64)
        else:
            # Empty constraints array with correct shape
            constraints = np.zeros((0, 3), dtype=np.float64)

        # Evaluate loads: accumulate nodal forces from both Force and Pressure loads
        # Key: node_index, Value: [fx, fy, fz] force components
        nodal_forces: dict[int, np.ndarray] = {}

        for query, load in self._state.loads:
            if isinstance(load, Force) and isinstance(query, NodeQuery):
                # Point force at nodes
                node_indices = query.evaluate(self._state.mesh, components)
                force_vec = np.array([load.fx, load.fy, load.fz])
                for node_idx in node_indices:
                    if node_idx not in nodal_forces:
                        nodal_forces[node_idx] = np.zeros(3)
                    nodal_forces[node_idx] += force_vec

            elif isinstance(load, Pressure) and isinstance(query, FaceQuery):
                # Pressure load on faces - convert to consistent nodal forces
                # Pressure acts in the negative normal direction (into the surface)
                face_indices = query.evaluate(self._state.mesh, components)

                for elem_idx, local_face_idx in face_indices:
                    # Get face properties
                    # Use get_face_all_nodes to include mid-edge nodes for quadratic elements
                    face_nodes = self._state.mesh.get_face_all_nodes(elem_idx, local_face_idx)
                    face_area = self._state.mesh.get_face_area(elem_idx, local_face_idx)
                    face_normal = self._state.mesh.get_face_normal(elem_idx, local_face_idx)

                    # Total force on face: F = pressure * area
                    # Direction: negative normal (pressure pushes into surface)
                    total_force = -load.value * face_area * face_normal

                    # Distribute force equally to all face nodes (consistent nodal forces
                    # for constant pressure on linear elements).
                    # Note: For quadratic elements, proper consistent nodal forces require
                    # integration with shape functions. Equal distribution is an approximation
                    # that works well for most practical cases with uniform pressure.
                    n_face_nodes = len(face_nodes)
                    force_per_node = total_force / n_face_nodes

                    for node_idx in face_nodes:
                        if node_idx not in nodal_forces:
                            nodal_forces[node_idx] = np.zeros(3)
                        nodal_forces[node_idx] += force_per_node

            elif isinstance(load, Force) and isinstance(query, FaceQuery):
                # Force applied to face - distribute to all face nodes
                face_indices = query.evaluate(self._state.mesh, components)
                force_vec = np.array([load.fx, load.fy, load.fz])

                for elem_idx, local_face_idx in face_indices:
                    # Use get_face_all_nodes to include mid-edge nodes for quadratic elements
                    face_nodes = self._state.mesh.get_face_all_nodes(elem_idx, local_face_idx)
                    n_face_nodes = len(face_nodes)
                    force_per_node = force_vec / n_face_nodes

                    for node_idx in face_nodes:
                        if node_idx not in nodal_forces:
                            nodal_forces[node_idx] = np.zeros(3)
                        nodal_forces[node_idx] += force_per_node

            elif isinstance(load, Pressure) and isinstance(query, NodeQuery):
                raise ValueError(
                    "Pressure loads must be applied to faces (FaceQuery), not nodes. "
                    "Use Faces.where(...) or Faces.on_boundary() to select faces."
                )

        # Convert nodal_forces dict to arrays for the solver
        if not nodal_forces:
            raise ValueError("No loads produced any nodal forces")

        # Build forces array: Nx4 where each row is [node_idx, fx, fy, fz]
        force_rows = []
        for node_idx, force_vec in nodal_forces.items():
            force_rows.append([float(node_idx), force_vec[0], force_vec[1], force_vec[2]])

        forces = np.array(force_rows, dtype=np.float64)

        # Determine thickness for 2D elements
        # Use the first assigned thickness, or default to 1.0
        if self._state.thickness_assignments:
            thickness = next(iter(self._state.thickness_assignments.values()))
        else:
            thickness = 1.0  # Default thickness for 2D plane stress elements

        # Pass the underlying Rust mesh to the solver
        mesh = self._state.mesh
        core_mesh = mesh._inner if hasattr(mesh, "_inner") else mesh

        # Choose solver based on material configuration
        if use_multi_material and len(self._state.materials) > 1:
            # Multi-material solve: pass all materials and element-material mapping
            materials_list = list(self._state.materials.values())

            # Build element_material_indices as Nx2 array
            elem_mat_rows = [
                [elem_idx, mat_idx]
                for elem_idx, mat_idx in element_material_map.items()
            ]
            if elem_mat_rows:
                element_material_indices = np.array(elem_mat_rows, dtype=np.int64)
            else:
                # No explicit assignments - use empty array (all elements get material 0)
                element_material_indices = np.zeros((0, 2), dtype=np.int64)

            return solve_with_materials(
                core_mesh,
                materials_list,
                element_material_indices,
                constraints,
                forces,
                config,
                thickness,
            )
        else:
            # Single-material solve (optimized path)
            return solve_with_forces(
                core_mesh,
                default_material,
                constraints,
                forces,
                config,
                thickness,
            )

    def __repr__(self) -> str:
        return (
            f"Model(mesh={self._state.mesh}, "
            f"materials={len(self._state.materials)}, "
            f"constraints={len(self._state.constraints)}, "
            f"loads={len(self._state.loads)})"
        )
