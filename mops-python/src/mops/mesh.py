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

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mops._core import Mesh as _CoreMesh

if TYPE_CHECKING:
    pass


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
    "tet4", "tet10", "hex8", "hex20",
    "tri3", "tri6", "quad4", "quad8",
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

        Args:
            model: A gmsh.model object with a generated mesh.
            element_type: Optional element type to extract. If None, uses the
                most common 3D element type found, or 2D if no 3D elements.
                Supported types: tet4, tet10, hex8, hex20, tri3, tri6, quad4, quad8.

        Returns:
            New Mesh instance containing the extracted mesh.

        Raises:
            MeshError: If no suitable elements found or mesh not generated.
            ImportError: If gmsh package is not installed.

        Example::

            import gmsh
            gmsh.initialize()
            gmsh.model.add("box")

            # Create geometry
            gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
            gmsh.model.occ.synchronize()

            # Generate mesh
            gmsh.model.mesh.generate(3)

            # Extract to mops
            from mops import Mesh
            mesh = Mesh.from_gmsh(gmsh.model)

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
        tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
        n_nodes = len(node_tags)

        # Reshape coordinates to Nx3
        nodes = np.array(node_coords, dtype=np.float64).reshape(n_nodes, 3)

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

            element_types, element_tags, node_tags_per_elem = model.mesh.getElements()

            # Find the matching element type
            found = False
            for i, etype in enumerate(element_types):
                if etype == gmsh_type_code:
                    elem_nodes = node_tags_per_elem[i]
                    found = True
                    break

            if not found:
                raise MeshError(
                    f"No {target_type} elements found in Gmsh model. "
                    f"Available element types: {[GMSH_ELEMENT_TYPES.get(t, ('unknown', 0))[0] for t in element_types]}"
                )

        else:
            # Auto-detect: prefer 3D elements, then 2D
            element_types, element_tags, node_tags_per_elem = model.mesh.getElements()

            # Priority order: 3D tetrahedral, 3D hex, 2D tri, 2D quad
            priority_order = [4, 11, 5, 12, 2, 9, 3, 10]  # tet4, tet10, hex8, hex20, tri3, tri6, quad4, quad8

            selected_type = None
            selected_idx = None
            selected_gmsh_code = None

            for ptype in priority_order:
                for i, etype in enumerate(element_types):
                    if etype == ptype:
                        selected_type = GMSH_ELEMENT_TYPES[ptype][0]
                        selected_idx = i
                        selected_gmsh_code = ptype
                        break
                if selected_type is not None:
                    break

            if selected_type is None:
                # Try to find any supported element type
                for i, etype in enumerate(element_types):
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
                    for t in element_types
                ]
                raise MeshError(
                    f"No supported element types found in Gmsh model. "
                    f"Available: {available}"
                )

            target_type = selected_type
            elem_nodes = node_tags_per_elem[selected_idx]

        # Get number of nodes per element
        n_nodes_per_elem = GMSH_ELEMENT_TYPES[gmsh_type_code if element_type else selected_gmsh_code][1]
        n_elements = len(elem_nodes) // n_nodes_per_elem

        # Convert Gmsh node tags to 0-based indices
        elements = np.zeros((n_elements, n_nodes_per_elem), dtype=np.int64)
        for i in range(n_elements):
            for j in range(n_nodes_per_elem):
                gmsh_tag = int(elem_nodes[i * n_nodes_per_elem + j])
                elements[i, j] = tag_to_idx[gmsh_tag]

        return cls.from_arrays(nodes, elements, target_type)

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
    def _core(self) -> _CoreMesh:
        """Access the underlying Rust Mesh (internal use only)."""
        return self._inner

    def __repr__(self) -> str:
        return f"Mesh(nodes={self.n_nodes}, elements={self.n_elements}, type={self.element_type})"
