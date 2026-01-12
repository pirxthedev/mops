"""Mesh module with Gmsh integration.

This module provides mesh creation and I/O functionality, wrapping the low-level
Rust Mesh implementation with Python conveniences like file loading and Gmsh support.

Example::

    from mops import Mesh
    import numpy as np

    # From numpy arrays
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    mesh = Mesh.from_arrays(nodes, elements, "tet4")

    # From Gmsh file
    mesh = Mesh.from_file("geometry.msh")

    # From live Gmsh model
    import gmsh
    gmsh.initialize()
    # ... build geometry ...
    gmsh.model.mesh.generate(3)
    mesh = Mesh.from_gmsh(gmsh.model)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mops._core import Mesh as _CoreMesh

if TYPE_CHECKING:
    pass


@dataclass
class PhysicalGroup:
    """A named physical group from Gmsh.

    Physical groups in Gmsh are named collections of geometric entities
    (points, curves, surfaces, volumes) that can be used for:
    - Applying boundary conditions to named surfaces
    - Assigning materials to named volumes
    - Extracting results for specific regions

    Attributes:
        name: User-defined name for this physical group.
        dimension: Topological dimension (0=point, 1=curve, 2=surface, 3=volume).
        tag: Gmsh physical group tag (integer identifier).
        node_indices: 0-based node indices belonging to this group.
        element_indices: 0-based element indices belonging to this group (for dim > 0).
    """

    name: str
    dimension: int
    tag: int
    node_indices: list[int] = field(default_factory=list)
    element_indices: list[int] = field(default_factory=list)


# Face definitions for each element type (CORNER NODES ONLY).
# Each entry is a tuple of node indices defining a face (counter-clockwise when viewed from outside).
# For 3D elements, faces are triangles or quads. For 2D elements, faces are edges.
# Using corner nodes only (linear topology for face identification and matching).
ELEMENT_FACES = {
    # 3D Tetrahedron: 4 triangular faces
    # Standard ordering: Face normals point outward when nodes are counter-clockwise
    "tet4": [
        (0, 2, 1),  # Face 0: base (nodes 0,1,2), normal points -z
        (0, 1, 3),  # Face 1: front face
        (1, 2, 3),  # Face 2: right face
        (2, 0, 3),  # Face 3: left face
    ],
    "tet10": [
        (0, 2, 1),  # Same corner topology as tet4
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
    ],
    # 3D Hexahedron: 6 quadrilateral faces
    # Standard brick ordering with nodes 0-3 on bottom (-z), 4-7 on top (+z)
    "hex8": [
        (0, 3, 2, 1),  # Face 0: bottom (-z)
        (4, 5, 6, 7),  # Face 1: top (+z)
        (0, 1, 5, 4),  # Face 2: front (-y)
        (2, 3, 7, 6),  # Face 3: back (+y)
        (0, 4, 7, 3),  # Face 4: left (-x)
        (1, 2, 6, 5),  # Face 5: right (+x)
    ],
    "hex8sri": [
        (0, 3, 2, 1),  # Same as hex8
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    ],
    "hex20": [
        (0, 3, 2, 1),  # Same corner topology as hex8
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    ],
    # 2D Triangle: 3 edge "faces" (for 2D, faces are edges)
    "tri3": [
        (0, 1),  # Edge 0
        (1, 2),  # Edge 1
        (2, 0),  # Edge 2
    ],
    "tri6": [
        (0, 1),  # Same corner topology as tri3
        (1, 2),
        (2, 0),
    ],
    # 2D Quadrilateral: 4 edge "faces"
    "quad4": [
        (0, 1),  # Edge 0: bottom
        (1, 2),  # Edge 1: right
        (2, 3),  # Edge 2: top
        (3, 0),  # Edge 3: left
    ],
    "quad8": [
        (0, 1),  # Same corner topology as quad4
        (1, 2),
        (2, 3),
        (3, 0),
    ],
}

# Complete face node definitions including mid-edge nodes for quadratic elements.
# Used for pressure loading where forces must be distributed to ALL face nodes.
# For linear elements, this is identical to ELEMENT_FACES.
# For quadratic elements, this includes mid-edge nodes on each face.
#
# Node ordering: corners first (same as ELEMENT_FACES), then mid-edge nodes
# traversing edges in order (corner0-corner1, corner1-corner2, etc.)
ELEMENT_FACE_ALL_NODES = {
    # Linear elements: same as ELEMENT_FACES
    "tet4": ELEMENT_FACES["tet4"],
    "hex8": ELEMENT_FACES["hex8"],
    "hex8sri": ELEMENT_FACES["hex8sri"],
    "tri3": ELEMENT_FACES["tri3"],
    "quad4": ELEMENT_FACES["quad4"],

    # Tet10: 6-node triangular faces (3 corners + 3 mid-edge)
    # Mid-edge node indices: 4=edge(0,1), 5=edge(1,2), 6=edge(0,2), 7=edge(0,3), 8=edge(1,3), 9=edge(2,3)
    "tet10": [
        (0, 2, 1, 6, 5, 4),  # Face 0: base - corners (0,2,1), edges 0-2, 2-1, 1-0 -> nodes 6, 5, 4
        (0, 1, 3, 4, 8, 7),  # Face 1: corners (0,1,3), edges 0-1, 1-3, 3-0 -> nodes 4, 8, 7
        (1, 2, 3, 5, 9, 8),  # Face 2: corners (1,2,3), edges 1-2, 2-3, 3-1 -> nodes 5, 9, 8
        (2, 0, 3, 6, 7, 9),  # Face 3: corners (2,0,3), edges 2-0, 0-3, 3-2 -> nodes 6, 7, 9
    ],

    # Hex20: 8-node quadrilateral faces (4 corners + 4 mid-edge)
    # Mid-edge node indices based on hex20 node numbering:
    #   8=edge(0,1), 9=edge(1,2), 10=edge(2,3), 11=edge(0,3) [bottom face edges]
    #   12=edge(0,4), 13=edge(1,5), 14=edge(2,6), 15=edge(3,7) [vertical edges]
    #   16=edge(4,5), 17=edge(5,6), 18=edge(6,7), 19=edge(4,7) [top face edges]
    "hex20": [
        # Face 0: bottom (-z), corners (0,3,2,1)
        # edges: 0-3->11, 3-2->10, 2-1->9, 1-0->8
        (0, 3, 2, 1, 11, 10, 9, 8),
        # Face 1: top (+z), corners (4,5,6,7)
        # edges: 4-5->16, 5-6->17, 6-7->18, 7-4->19
        (4, 5, 6, 7, 16, 17, 18, 19),
        # Face 2: front (-y), corners (0,1,5,4)
        # edges: 0-1->8, 1-5->13, 5-4->16, 4-0->12
        (0, 1, 5, 4, 8, 13, 16, 12),
        # Face 3: back (+y), corners (2,3,7,6)
        # edges: 2-3->10, 3-7->15, 7-6->18, 6-2->14
        (2, 3, 7, 6, 10, 15, 18, 14),
        # Face 4: left (-x), corners (0,4,7,3)
        # edges: 0-4->12, 4-7->19, 7-3->15, 3-0->11
        (0, 4, 7, 3, 12, 19, 15, 11),
        # Face 5: right (+x), corners (1,2,6,5)
        # edges: 1-2->9, 2-6->14, 6-5->17, 5-1->13
        (1, 2, 6, 5, 9, 14, 17, 13),
    ],

    # Tri6: 3-node edges (2 corners + 1 mid-edge)
    # Mid-edge nodes: 3=edge(0,1), 4=edge(1,2), 5=edge(0,2)
    "tri6": [
        (0, 1, 3),  # Edge 0: nodes 0, 1 + midpoint 3
        (1, 2, 4),  # Edge 1: nodes 1, 2 + midpoint 4
        (2, 0, 5),  # Edge 2: nodes 2, 0 + midpoint 5
    ],

    # Quad8: 3-node edges (2 corners + 1 mid-edge)
    # Mid-edge nodes: 4=edge(0,1), 5=edge(1,2), 6=edge(2,3), 7=edge(3,0)
    "quad8": [
        (0, 1, 4),  # Edge 0: bottom
        (1, 2, 5),  # Edge 1: right
        (2, 3, 6),  # Edge 2: top
        (3, 0, 7),  # Edge 3: left
    ],
}


# Edge definitions for each element type.
# Each entry is a tuple of (node_index_1, node_index_2) defining an edge.
# Only corner nodes are used for edges (linear topology).
ELEMENT_EDGES = {
    # 3D elements
    "tet4": [
        (0, 1), (1, 2), (2, 0),  # base triangle
        (0, 3), (1, 3), (2, 3),  # edges to apex
    ],
    "tet10": [
        (0, 1), (1, 2), (2, 0),  # base triangle (using corner nodes only)
        (0, 3), (1, 3), (2, 3),  # edges to apex
    ],
    "hex8": [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ],
    "hex8sri": [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Same as hex8
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ],
    "hex20": [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face (corner nodes only)
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ],
    # 2D elements
    "tri3": [
        (0, 1), (1, 2), (2, 0),  # triangle edges
    ],
    "tri6": [
        (0, 1), (1, 2), (2, 0),  # triangle edges (corner nodes only)
    ],
    "quad4": [
        (0, 1), (1, 2), (2, 3), (3, 0),  # quad edges
    ],
    "quad8": [
        (0, 1), (1, 2), (2, 3), (3, 0),  # quad edges (corner nodes only)
    ],
}


# Mapping from Gmsh element type codes to mops element types.
# Reference: Gmsh documentation, section 9.1 "MSH file format"
# https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
GMSH_ELEMENT_TYPES = {
    # 1D elements (not supported yet)
    1: ("line2", 2),
    8: ("line3", 3),
    # 2D elements
    2: ("tri3", 3),
    3: ("quad4", 4),
    9: ("tri6", 6),
    10: ("quad8", 8),
    16: ("quad9", 9),  # Not supported
    # 3D elements
    4: ("tet4", 4),
    5: ("hex8", 8),
    6: ("prism6", 6),  # Not supported
    7: ("pyramid5", 5),  # Not supported
    11: ("tet10", 10),
    12: ("hex20", 20),
    17: ("hex27", 27),  # Not supported
    18: ("prism15", 15),  # Not supported
    19: ("prism18", 18),  # Not supported
    29: ("tet20", 20),  # Not supported (different from tet10)
}

# Element types supported in mops-core
SUPPORTED_ELEMENT_TYPES = frozenset([
    "tet4", "tet10", "hex8", "hex8sri", "hex8bbar", "hex20",
    "tri3", "tri6", "quad4", "quad8",
    "tri3axisymmetric", "quad4axisymmetric",
])


class MeshError(Exception):
    """Error raised for mesh-related issues."""
    pass


class Mesh:
    """Finite element mesh with nodes and elements.

    The Mesh class provides a high-level interface for creating and manipulating
    finite element meshes. It wraps the low-level Rust implementation and adds
    Python conveniences like file I/O and Gmsh integration.

    Meshes are immutable after creation. All element types within a single Mesh
    must be the same (homogeneous mesh).

    Attributes:
        n_nodes: Number of nodes in the mesh.
        n_elements: Number of elements in the mesh.
        element_type: Type of elements in this mesh (e.g., "tet4", "hex8").
        coords: Nx3 array of node coordinates.

    Example::

        # Create from arrays (preferred method)
        mesh = Mesh.from_arrays(nodes, elements, "tet4")

        # Direct constructor also works (backwards compatible)
        mesh = Mesh(nodes, elements, "tet4")

        # Create from Gmsh file
        mesh = Mesh.from_file("model.msh")

        # Access properties
        print(f"Mesh has {mesh.n_nodes} nodes and {mesh.n_elements} elements")
    """

    def __init__(
        self,
        nodes_or_inner: NDArray[np.float64] | _CoreMesh,
        elements_or_type: NDArray[np.int64] | str | None = None,
        element_type: str | None = None,
        *,
        physical_groups: dict[str, PhysicalGroup] | None = None,
    ) -> None:
        """Initialize Mesh from arrays or internal core mesh.

        Can be called in two ways:
        1. Mesh(nodes, elements, element_type) - create from arrays
        2. Mesh(core_mesh, element_type) - internal, wrap existing core mesh

        For clarity, prefer using factory methods like from_arrays(), from_file(),
        or from_gmsh() to create Mesh instances.

        Args:
            nodes_or_inner: Either Nx3 array of node coordinates, or internal core mesh.
            elements_or_type: Either MxK array of element connectivity, or element type string.
            element_type: Element type string (only when using array form).
            physical_groups: Optional dict mapping group names to PhysicalGroup objects.
        """
        # Detect which calling convention is being used
        if isinstance(nodes_or_inner, _CoreMesh):
            # Internal constructor: Mesh(core_mesh, element_type)
            self._inner = nodes_or_inner
            self._element_type = elements_or_type if isinstance(elements_or_type, str) else element_type
        else:
            # Array constructor: Mesh(nodes, elements, element_type)
            nodes = np.asarray(nodes_or_inner, dtype=np.float64)
            elements = np.asarray(elements_or_type, dtype=np.int64)

            if element_type is None:
                raise ValueError("element_type is required when creating mesh from arrays")

            if element_type not in SUPPORTED_ELEMENT_TYPES:
                raise ValueError(
                    f"Unsupported element type: {element_type}. "
                    f"Supported types: {sorted(SUPPORTED_ELEMENT_TYPES)}"
                )

            if nodes.ndim != 2 or nodes.shape[1] != 3:
                raise ValueError(f"nodes must be Nx3 array, got shape {nodes.shape}")

            self._inner = _CoreMesh(nodes, elements, element_type)
            self._element_type = element_type

        # Store physical groups (empty dict if none provided)
        self._physical_groups: dict[str, PhysicalGroup] = physical_groups or {}

    @classmethod
    def from_arrays(
        cls,
        nodes: NDArray[np.float64],
        elements: NDArray[np.int64],
        element_type: str,
    ) -> Mesh:
        """Create mesh from numpy arrays.

        Args:
            nodes: Nx3 array of node coordinates (x, y, z).
            elements: MxK array of element connectivity, where K is the number
                of nodes per element. Node indices are 0-based.
            element_type: Element type string. Supported types:
                - 3D: "tet4", "tet10", "hex8", "hex20"
                - 2D: "tri3", "tri6", "quad4", "quad8"

        Returns:
            New Mesh instance.

        Raises:
            ValueError: If arrays have wrong shape or element_type is unknown.
            MeshError: If element connectivity references invalid node indices.

        Example::

            nodes = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=np.float64)

            elements = np.array([[0, 1, 2, 3]], dtype=np.int64)

            mesh = Mesh.from_arrays(nodes, elements, "tet4")
        """
        # Delegate to constructor - validation happens there
        return cls(nodes, elements, element_type)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        element_type: str | None = None,
    ) -> Mesh:
        """Load mesh from a file.

        Supports Gmsh MSH format (versions 2.2 and 4.x). For other formats,
        use Gmsh to convert first.

        Args:
            path: Path to mesh file (.msh).
            element_type: Optional element type to extract. If None, uses the
                most common 3D element type found, or 2D if no 3D elements.

        Returns:
            New Mesh instance.

        Raises:
            FileNotFoundError: If file does not exist.
            MeshError: If file format is unsupported or parsing fails.
            ImportError: If gmsh package is not installed.

        Example::

            mesh = Mesh.from_file("geometry.msh")
            mesh = Mesh.from_file("model.msh", element_type="tet10")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in (".msh",):
            raise MeshError(
                f"Unsupported file format: {suffix}. "
                "Supported formats: .msh (Gmsh)"
            )

        try:
            import gmsh
        except ImportError:
            raise ImportError(
                "gmsh package required for mesh file loading. "
                "Install with: pip install gmsh"
            )

        # Initialize Gmsh and load the file
        needs_finalize = False
        if not gmsh.isInitialized():
            gmsh.initialize()
            needs_finalize = True

        try:
            gmsh.option.setNumber("General.Terminal", 0)  # Suppress output
            gmsh.open(str(path))
            mesh = cls.from_gmsh(gmsh.model, element_type=element_type)
            gmsh.clear()
            return mesh
        finally:
            if needs_finalize:
                gmsh.finalize()

    @classmethod
    def from_gmsh(
        cls,
        model: object,  # gmsh.model type, but can't import at runtime
        element_type: str | None = None,
    ) -> Mesh:
        """Create mesh from a Gmsh model.

        This method extracts mesh data from an active Gmsh model. The model
        must already have a generated mesh (call gmsh.model.mesh.generate()
        before this method).

        Physical groups defined in Gmsh are extracted and made available
        through the mesh.physical_groups property. This allows applying
        boundary conditions to named surfaces/regions.

        Args:
            model: A gmsh.model object with a generated mesh.
            element_type: Optional element type to extract. If None, uses the
                most common 3D element type found, or 2D if no 3D elements.
                Supported types: tet4, tet10, hex8, hex20, tri3, tri6, quad4, quad8.

        Returns:
            New Mesh instance containing the extracted mesh and physical groups.

        Raises:
            MeshError: If no suitable elements found or mesh not generated.
            ImportError: If gmsh package is not installed.

        Example::

            import gmsh
            gmsh.initialize()
            gmsh.model.add("box")

            # Create geometry
            box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
            gmsh.model.occ.synchronize()

            # Define physical groups for boundaries
            surfaces = gmsh.model.getBoundary([(3, box)], oriented=False)
            bottom = [s for s in surfaces if abs(gmsh.model.getBoundingBox(2, s[1])[2]) < 1e-6]
            gmsh.model.addPhysicalGroup(2, [s[1] for s in bottom], name="fixed")

            # Generate mesh
            gmsh.model.mesh.generate(3)

            # Extract to mops
            from mops import Mesh
            mesh = Mesh.from_gmsh(gmsh.model)

            # Access physical groups
            fixed = mesh.get_physical_group("fixed")
            print(f"Fixed boundary has {len(fixed.node_indices)} nodes")

            gmsh.finalize()
        """
        try:
            import gmsh
        except ImportError:
            raise ImportError(
                "gmsh package required for Gmsh integration. "
                "Install with: pip install gmsh"
            )

        # Get all nodes
        node_tags, node_coords, _ = model.mesh.getNodes()
        if len(node_tags) == 0:
            raise MeshError(
                "No nodes found in Gmsh model. "
                "Did you call gmsh.model.mesh.generate()?"
            )

        # Build node tag to index mapping (Gmsh tags are 1-based and may have gaps)
        tag_to_idx = {int(tag): idx for idx, tag in enumerate(node_tags)}
        n_nodes = len(node_tags)

        # Reshape coordinates to Nx3
        nodes = np.array(node_coords, dtype=np.float64).reshape(n_nodes, 3)

        # Get all elements once for both element extraction and physical group mapping
        all_element_types, all_element_tags, all_node_tags_per_elem = model.mesh.getElements()

        # Build element tag to index mapping
        # We need to map Gmsh element tags to our 0-based element indices
        # Gmsh element tags are unique across all element types
        elem_tag_to_idx: dict[int, int] = {}
        final_gmsh_type_code: int | None = None

        # Determine which element type to extract
        if element_type is not None:
            # User specified type - find corresponding Gmsh type
            target_type = element_type.lower()
            if target_type not in SUPPORTED_ELEMENT_TYPES:
                raise ValueError(
                    f"Unsupported element type: {target_type}. "
                    f"Supported: {sorted(SUPPORTED_ELEMENT_TYPES)}"
                )

            # Find Gmsh element type code for this type
            gmsh_type_code = None
            for code, (name, _) in GMSH_ELEMENT_TYPES.items():
                if name == target_type:
                    gmsh_type_code = code
                    break

            if gmsh_type_code is None:
                raise MeshError(f"Cannot map element type '{target_type}' to Gmsh code")

            # Find the matching element type
            found = False
            for i, etype in enumerate(all_element_types):
                if etype == gmsh_type_code:
                    elem_nodes = all_node_tags_per_elem[i]
                    elem_tags = all_element_tags[i]
                    found = True
                    break

            if not found:
                raise MeshError(
                    f"No {target_type} elements found in Gmsh model. "
                    f"Available element types: {[GMSH_ELEMENT_TYPES.get(t, ('unknown', 0))[0] for t in all_element_types]}"
                )

            final_gmsh_type_code = gmsh_type_code

        else:
            # Auto-detect: prefer 3D elements, then 2D
            # Priority order: 3D tetrahedral, 3D hex, 2D tri, 2D quad
            priority_order = [4, 11, 5, 12, 2, 9, 3, 10]  # tet4, tet10, hex8, hex20, tri3, tri6, quad4, quad8

            selected_type = None
            selected_idx = None
            selected_gmsh_code = None

            for ptype in priority_order:
                for i, etype in enumerate(all_element_types):
                    if etype == ptype:
                        selected_type = GMSH_ELEMENT_TYPES[ptype][0]
                        selected_idx = i
                        selected_gmsh_code = ptype
                        break
                if selected_type is not None:
                    break

            if selected_type is None:
                # Try to find any supported element type
                for i, etype in enumerate(all_element_types):
                    if etype in GMSH_ELEMENT_TYPES:
                        type_name, _ = GMSH_ELEMENT_TYPES[etype]
                        if type_name in SUPPORTED_ELEMENT_TYPES:
                            selected_type = type_name
                            selected_idx = i
                            selected_gmsh_code = etype
                            break

            if selected_type is None:
                available = [
                    GMSH_ELEMENT_TYPES.get(t, (f"gmsh_{t}", 0))[0]
                    for t in all_element_types
                ]
                raise MeshError(
                    f"No supported element types found in Gmsh model. "
                    f"Available: {available}"
                )

            target_type = selected_type
            elem_nodes = all_node_tags_per_elem[selected_idx]
            elem_tags = all_element_tags[selected_idx]
            final_gmsh_type_code = selected_gmsh_code

        # Get number of nodes per element
        n_nodes_per_elem = GMSH_ELEMENT_TYPES[final_gmsh_type_code][1]
        n_elements = len(elem_nodes) // n_nodes_per_elem

        # Build element tag to 0-based index mapping for primary elements
        for i, tag in enumerate(elem_tags):
            elem_tag_to_idx[int(tag)] = i

        # Convert Gmsh node tags to 0-based indices
        elements = np.zeros((n_elements, n_nodes_per_elem), dtype=np.int64)
        for i in range(n_elements):
            for j in range(n_nodes_per_elem):
                gmsh_tag = int(elem_nodes[i * n_nodes_per_elem + j])
                elements[i, j] = tag_to_idx[gmsh_tag]

        # Extract physical groups
        physical_groups: dict[str, PhysicalGroup] = {}

        try:
            # Get all physical groups (returns list of (dim, tag) tuples)
            phys_groups = model.getPhysicalGroups()

            for dim, phys_tag in phys_groups:
                # Get the name of this physical group
                name = model.getPhysicalName(dim, phys_tag)
                if not name:
                    # Generate a name if none provided (use dimension and tag)
                    name = f"dim{dim}_tag{phys_tag}"

                # Get entities (surfaces, volumes, etc.) in this physical group
                entities = model.getEntitiesForPhysicalGroup(dim, phys_tag)

                # Collect node indices
                node_indices_set: set[int] = set()
                element_indices_set: set[int] = set()

                for entity_tag in entities:
                    # Get nodes for this entity
                    try:
                        entity_node_tags, _, _ = model.mesh.getNodes(dim, entity_tag)
                        for ntag in entity_node_tags:
                            if int(ntag) in tag_to_idx:
                                node_indices_set.add(tag_to_idx[int(ntag)])
                    except Exception:
                        # Entity might not have nodes, continue
                        pass

                    # Get elements for this entity
                    # Only include elements that match our target element type
                    try:
                        ent_elem_types, ent_elem_tags, _ = model.mesh.getElements(dim, entity_tag)
                        for i, etype in enumerate(ent_elem_types):
                            if etype == final_gmsh_type_code:
                                for etag in ent_elem_tags[i]:
                                    if int(etag) in elem_tag_to_idx:
                                        element_indices_set.add(elem_tag_to_idx[int(etag)])
                    except Exception:
                        # Entity might not have elements, continue
                        pass

                # Create the physical group
                physical_groups[name] = PhysicalGroup(
                    name=name,
                    dimension=dim,
                    tag=phys_tag,
                    node_indices=sorted(node_indices_set),
                    element_indices=sorted(element_indices_set),
                )

        except Exception:
            # If physical group extraction fails, continue with empty groups
            # This maintains backwards compatibility with meshes without physical groups
            pass

        # Create mesh with physical groups
        mesh = cls(nodes, elements, target_type, physical_groups=physical_groups)
        return mesh

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the mesh."""
        return self._inner.n_nodes

    @property
    def n_elements(self) -> int:
        """Number of elements in the mesh."""
        return self._inner.n_elements

    @property
    def element_type(self) -> str:
        """Element type (e.g., 'tet4', 'hex8')."""
        return self._element_type

    @property
    def coords(self) -> NDArray[np.float64]:
        """Node coordinates as Nx3 array."""
        return self._inner.coords

    @property
    def elements(self) -> NDArray[np.int64]:
        """Element connectivity as MxK array (node indices)."""
        return self._inner.elements

    @property
    def _core(self) -> _CoreMesh:
        """Access the underlying Rust Mesh (internal use only)."""
        return self._inner

    @property
    def physical_groups(self) -> dict[str, PhysicalGroup]:
        """Physical groups extracted from Gmsh.

        Returns a dictionary mapping group names to PhysicalGroup objects.
        Empty dict if mesh was not created from Gmsh or if no physical groups
        were defined in the Gmsh model.

        Example::

            mesh = Mesh.from_gmsh(gmsh.model)
            for name, group in mesh.physical_groups.items():
                print(f"{name}: {len(group.node_indices)} nodes")
        """
        return self._physical_groups

    def get_physical_group(self, name: str) -> PhysicalGroup:
        """Get a physical group by name.

        Args:
            name: Name of the physical group.

        Returns:
            PhysicalGroup object containing node and element indices.

        Raises:
            KeyError: If no physical group with this name exists.

        Example::

            mesh = Mesh.from_gmsh(gmsh.model)
            fixed_face = mesh.get_physical_group("fixed")
            # Use node indices for boundary conditions
            fixed_nodes = fixed_face.node_indices
        """
        if name not in self._physical_groups:
            available = list(self._physical_groups.keys())
            raise KeyError(
                f"Unknown physical group: '{name}'. "
                f"Available groups: {available}"
            )
        return self._physical_groups[name]

    def list_physical_groups(self) -> list[str]:
        """List names of all physical groups.

        Returns:
            List of physical group names.
        """
        return list(self._physical_groups.keys())

    def plot(
        self,
        filename: str | Path | None = None,
        *,
        show_nodes: bool = True,
        show_edges: bool = True,
        show_labels: bool = False,
        node_size: float = 20,
        edge_color: str = "black",
        node_color: str = "blue",
        figsize: tuple[float, float] = (8, 8),
        dpi: int = 100,
        elev: float = 30,
        azim: float = 45,
    ) -> bytes | None:
        """Plot mesh geometry for visualization.

        This method renders the mesh as a 3D wireframe plot showing nodes
        and element edges. Useful for verifying mesh creation and geometry.

        Args:
            filename: If provided, save the plot to this file path. If None,
                returns the plot as PNG bytes (useful for LLM consumption).
            show_nodes: Whether to show node markers (default True).
            show_edges: Whether to show element edges (default True).
            show_labels: Whether to show node/element index labels (default False).
            node_size: Size of node markers in points (default 20).
            edge_color: Color of element edges (default "black").
            node_color: Color of node markers (default "blue").
            figsize: Figure size in inches as (width, height) (default (8, 8)).
            dpi: Resolution in dots per inch (default 100).
            elev: Elevation angle in degrees for 3D view (default 30).
            azim: Azimuth angle in degrees for 3D view (default 45).

        Returns:
            If filename is None, returns PNG image as bytes.
            If filename is provided, saves to file and returns None.

        Raises:
            ImportError: If matplotlib is not installed.

        Example::

            mesh = Mesh.from_arrays(nodes, elements, "tet4")

            # Get PNG bytes for display
            png_bytes = mesh.plot()

            # Save to file
            mesh.plot("my_mesh.png")

            # Customize appearance
            mesh.plot(show_labels=True, node_color="red", elev=60, azim=30)
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Headless backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError:
            raise ImportError(
                "matplotlib required for mesh plotting. "
                "Install with: pip install matplotlib"
            )

        coords = self.coords
        elements = self.elements
        elem_type = self.element_type

        # Determine if 2D or 3D
        is_2d = elem_type in ("tri3", "tri6", "quad4", "quad8")

        fig = plt.figure(figsize=figsize, dpi=dpi)

        if is_2d:
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")
        else:
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=elev, azim=azim)

        # Draw edges
        if show_edges and elem_type in ELEMENT_EDGES:
            edge_defs = ELEMENT_EDGES[elem_type]
            for elem in elements:
                for i1, i2 in edge_defs:
                    n1, n2 = elem[i1], elem[i2]
                    if is_2d:
                        ax.plot(
                            [coords[n1, 0], coords[n2, 0]],
                            [coords[n1, 1], coords[n2, 1]],
                            color=edge_color,
                            linewidth=0.8,
                        )
                    else:
                        ax.plot(
                            [coords[n1, 0], coords[n2, 0]],
                            [coords[n1, 1], coords[n2, 1]],
                            [coords[n1, 2], coords[n2, 2]],
                            color=edge_color,
                            linewidth=0.8,
                        )

        # Draw nodes
        if show_nodes:
            if is_2d:
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    s=node_size,
                    c=node_color,
                    zorder=5,
                )
            else:
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    s=node_size,
                    c=node_color,
                )

        # Draw labels
        if show_labels:
            for i, coord in enumerate(coords):
                if is_2d:
                    ax.annotate(
                        str(i),
                        (coord[0], coord[1]),
                        textcoords="offset points",
                        xytext=(3, 3),
                        fontsize=8,
                    )
                else:
                    ax.text(coord[0], coord[1], coord[2], str(i), fontsize=8)

        # Labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if not is_2d:
            ax.set_zlabel("Z")

        ax.set_title(f"Mesh: {self.n_nodes} nodes, {self.n_elements} {elem_type} elements")

        plt.tight_layout()

        # Return bytes or save to file
        if filename is None:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=dpi)
            plt.close(fig)
            buf.seek(0)
            return buf.read()
        else:
            plt.savefig(filename, dpi=dpi)
            plt.close(fig)
            return None

    def get_all_faces(self) -> np.ndarray:
        """Get all faces in the mesh as (element_idx, local_face_idx) pairs.

        Returns:
            Nx2 array where each row is [element_index, local_face_index].
        """
        if not hasattr(self, "_all_faces"):
            elem_type = self.element_type
            if elem_type not in ELEMENT_FACES:
                raise MeshError(f"Face extraction not supported for element type: {elem_type}")

            n_faces_per_elem = len(ELEMENT_FACES[elem_type])
            n_elements = self.n_elements

            # Create array of all (element_idx, local_face_idx) pairs
            faces = np.zeros((n_elements * n_faces_per_elem, 2), dtype=np.int64)
            for i in range(n_elements):
                for j in range(n_faces_per_elem):
                    faces[i * n_faces_per_elem + j] = [i, j]

            self._all_faces = faces
        return self._all_faces

    def get_boundary_faces(self) -> np.ndarray:
        """Get boundary faces (faces not shared between elements).

        A face is on the boundary if it appears only once in the mesh
        (not shared by two elements).

        Returns:
            Nx2 array of boundary face indices [element_idx, local_face_idx].
        """
        if not hasattr(self, "_boundary_faces"):
            elem_type = self.element_type
            if elem_type not in ELEMENT_FACES:
                raise MeshError(f"Face extraction not supported for element type: {elem_type}")

            face_defs = ELEMENT_FACES[elem_type]
            elements = self.elements

            # Build a dictionary mapping face node sets to (element_idx, local_face_idx)
            # A boundary face appears exactly once
            face_count: dict[frozenset[int], list[tuple[int, int]]] = {}

            for elem_idx, elem in enumerate(elements):
                for local_face_idx, face_def in enumerate(face_defs):
                    # Get global node indices for this face
                    face_nodes = frozenset(int(elem[n]) for n in face_def)
                    if face_nodes not in face_count:
                        face_count[face_nodes] = []
                    face_count[face_nodes].append((elem_idx, local_face_idx))

            # Boundary faces are those that appear exactly once
            boundary = []
            for face_nodes, occurrences in face_count.items():
                if len(occurrences) == 1:
                    boundary.append(occurrences[0])

            self._boundary_faces = np.array(boundary, dtype=np.int64).reshape(-1, 2)
        return self._boundary_faces

    def get_face_nodes(self, element_idx: int, local_face_idx: int) -> np.ndarray:
        """Get global node indices for a specific face (corner nodes only).

        This returns only corner nodes, which is sufficient for face identification
        and geometry computation (centroid, normal, area). For pressure loading
        on quadratic elements, use get_face_all_nodes() instead.

        Args:
            element_idx: Element index.
            local_face_idx: Local face index within the element.

        Returns:
            Array of global node indices forming this face (corners only).
        """
        elem_type = self.element_type
        if elem_type not in ELEMENT_FACES:
            raise MeshError(f"Face nodes not defined for element type: {elem_type}")

        face_def = ELEMENT_FACES[elem_type][local_face_idx]
        elem = self.elements[element_idx]
        return np.array([elem[n] for n in face_def], dtype=np.int64)

    def get_face_all_nodes(self, element_idx: int, local_face_idx: int) -> np.ndarray:
        """Get ALL global node indices for a face, including mid-edge nodes.

        For linear elements (tet4, hex8, tri3, quad4), this returns the same nodes
        as get_face_nodes(). For quadratic elements (tet10, hex20, tri6, quad8),
        this includes the mid-edge nodes on the face.

        This method should be used when distributing pressure loads to ensure
        forces are applied to all face nodes with proper shape function weighting.

        Args:
            element_idx: Element index.
            local_face_idx: Local face index within the element.

        Returns:
            Array of global node indices for all nodes on this face.
            For quadratic elements, corners come first, then mid-edge nodes.
        """
        elem_type = self.element_type
        if elem_type not in ELEMENT_FACE_ALL_NODES:
            raise MeshError(f"Face nodes not defined for element type: {elem_type}")

        face_def = ELEMENT_FACE_ALL_NODES[elem_type][local_face_idx]
        elem = self.elements[element_idx]
        return np.array([elem[n] for n in face_def], dtype=np.int64)

    def get_face_centroid(self, element_idx: int, local_face_idx: int) -> np.ndarray:
        """Compute centroid of a face.

        Args:
            element_idx: Element index.
            local_face_idx: Local face index within the element.

        Returns:
            3D coordinates of face centroid.
        """
        face_nodes = self.get_face_nodes(element_idx, local_face_idx)
        coords = self.coords
        return np.mean(coords[face_nodes], axis=0)

    def get_face_normal(self, element_idx: int, local_face_idx: int) -> np.ndarray:
        """Compute outward unit normal vector for a face.

        For 3D elements (triangular/quad faces), computes cross product normal.
        For 2D elements (edge faces), computes perpendicular in xy-plane.

        Args:
            element_idx: Element index.
            local_face_idx: Local face index within the element.

        Returns:
            Unit normal vector (3D array).
        """
        face_nodes = self.get_face_nodes(element_idx, local_face_idx)
        coords = self.coords

        if len(face_nodes) >= 3:
            # 3D face (triangle or quad): use cross product
            p0 = coords[face_nodes[0]]
            p1 = coords[face_nodes[1]]
            p2 = coords[face_nodes[2]]
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
        elif len(face_nodes) == 2:
            # 2D edge: rotate edge vector 90 degrees in xy-plane
            p0 = coords[face_nodes[0]]
            p1 = coords[face_nodes[1]]
            edge = p1 - p0
            # Perpendicular in xy-plane: (-dy, dx, 0)
            normal = np.array([-edge[1], edge[0], 0.0])
        else:
            raise MeshError(f"Invalid face with {len(face_nodes)} nodes")

        # Normalize
        norm = np.linalg.norm(normal)
        if norm < 1e-14:
            raise MeshError("Degenerate face with zero area")
        return normal / norm

    def get_face_area(self, element_idx: int, local_face_idx: int) -> float:
        """Compute area of a face.

        For 3D elements with triangular faces: area = 0.5 * |v1 x v2|
        For 3D elements with quadrilateral faces: split into triangles and sum.
        For 2D elements (edges): returns edge length (for pressure = force/length).

        Args:
            element_idx: Element index.
            local_face_idx: Local face index within the element.

        Returns:
            Face area (or edge length for 2D elements).
        """
        face_nodes = self.get_face_nodes(element_idx, local_face_idx)
        coords = self.coords

        n_nodes = len(face_nodes)

        if n_nodes == 3:
            # Triangle: area = 0.5 * |v1 x v2|
            p0 = coords[face_nodes[0]]
            p1 = coords[face_nodes[1]]
            p2 = coords[face_nodes[2]]
            v1 = p1 - p0
            v2 = p2 - p0
            cross = np.cross(v1, v2)
            return 0.5 * np.linalg.norm(cross)

        elif n_nodes == 4:
            # Quadrilateral: split into two triangles (0-1-2 and 0-2-3)
            p0 = coords[face_nodes[0]]
            p1 = coords[face_nodes[1]]
            p2 = coords[face_nodes[2]]
            p3 = coords[face_nodes[3]]

            # Triangle 0-1-2
            v1 = p1 - p0
            v2 = p2 - p0
            area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))

            # Triangle 0-2-3
            v3 = p3 - p0
            area2 = 0.5 * np.linalg.norm(np.cross(v2, v3))

            return area1 + area2

        elif n_nodes == 2:
            # Edge (2D): return length
            p0 = coords[face_nodes[0]]
            p1 = coords[face_nodes[1]]
            return np.linalg.norm(p1 - p0)

        else:
            raise MeshError(f"Unsupported face with {n_nodes} nodes")

    def get_all_face_centroids(self) -> np.ndarray:
        """Get centroids for all faces in the mesh.

        Returns:
            Nx3 array of face centroids, ordered by (element_idx, local_face_idx).
        """
        if not hasattr(self, "_face_centroids"):
            all_faces = self.get_all_faces()
            centroids = np.zeros((len(all_faces), 3), dtype=np.float64)
            for i, (elem_idx, local_face_idx) in enumerate(all_faces):
                centroids[i] = self.get_face_centroid(elem_idx, local_face_idx)
            self._face_centroids = centroids
        return self._face_centroids

    def get_all_face_normals(self) -> np.ndarray:
        """Get unit normals for all faces in the mesh.

        Returns:
            Nx3 array of unit normal vectors, ordered by (element_idx, local_face_idx).
        """
        if not hasattr(self, "_face_normals"):
            all_faces = self.get_all_faces()
            normals = np.zeros((len(all_faces), 3), dtype=np.float64)
            for i, (elem_idx, local_face_idx) in enumerate(all_faces):
                normals[i] = self.get_face_normal(elem_idx, local_face_idx)
            self._face_normals = normals
        return self._face_normals

    def __repr__(self) -> str:
        return f"Mesh(nodes={self.n_nodes}, elements={self.n_elements}, type={self.element_type})"
