"""Visualization module with PyVista integration.

This module provides visualization capabilities for MOPS results using PyVista.
It supports both interactive visualization in Jupyter notebooks and export to
VTU format for ParaView.

Example::

    from mops import solve, Model

    # Solve model
    results = solve(model)

    # Interactive plot (works in Jupyter)
    results.plot("von_mises")

    # Export to VTU for ParaView
    results.export("results.vtu")

    # Or using the viz module directly
    from mops.viz import plot_results, export_vtu

    plot_results(results, model, field="displacement_magnitude")
    export_vtu(results, model, "output.vtu")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from mops.model import Model
    from mops.results import Results


# Map element types to PyVista/VTK cell types
# Reference: https://vtk.org/doc/nightly/html/vtkCellType_8h.html
VTK_CELL_TYPES = {
    # 3D elements
    "tet4": 10,     # VTK_TETRA
    "tet10": 24,    # VTK_QUADRATIC_TETRA
    "hex8": 12,     # VTK_HEXAHEDRON
    "hex8sri": 12,  # VTK_HEXAHEDRON (same as hex8, just different integration)
    "hex8bbar": 12, # VTK_HEXAHEDRON (same as hex8, just different integration)
    "hex20": 25,    # VTK_QUADRATIC_HEXAHEDRON
    # 2D elements
    "tri3": 5,      # VTK_TRIANGLE
    "tri6": 22,     # VTK_QUADRATIC_TRIANGLE
    "quad4": 9,     # VTK_QUAD
    "quad8": 23,    # VTK_QUADRATIC_QUAD
}

# Default colormaps for different field types
DEFAULT_COLORMAPS = {
    "von_mises": "jet",
    "displacement": "viridis",
    "displacement_magnitude": "viridis",
    "stress": "jet",
    "principal": "coolwarm",
    "tresca": "jet",
    "hydrostatic": "RdBu_r",
}


FieldName = Literal[
    "displacement",
    "displacement_magnitude",
    "displacement_x",
    "displacement_y",
    "displacement_z",
    "von_mises",
    "stress_xx",
    "stress_yy",
    "stress_zz",
    "stress_xy",
    "stress_yz",
    "stress_xz",
    "principal_1",
    "principal_2",
    "principal_3",
    "tresca",
    "hydrostatic",
]


def _check_pyvista() -> None:
    """Check if PyVista is available."""
    try:
        import pyvista  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyvista package required for visualization. "
            "Install with: pip install pyvista"
        )


def _get_mesh_from_model_or_results(
    results: "Results",
    model: "Model | None" = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Get mesh data from model or loaded results.

    Args:
        results: Results object.
        model: Optional model (required for in-memory results).

    Returns:
        Tuple of (nodes, elements, element_type).

    Raises:
        ValueError: If mesh data not available.
    """
    # Try to get from model first
    if model is not None:
        mesh = model.mesh
        return mesh.coords, mesh.elements, mesh.element_type

    # Try to get from HDF5-loaded results
    if results._lazy and results._h5file is not None:
        h5 = results._h5file
        if "/mesh/nodes" in h5 and "/mesh/elements" in h5:
            nodes = h5["/mesh/nodes"][:]
            elements = h5["/mesh/elements"][:]
            element_type = h5["/mesh/element_type"][()].decode()
            return nodes, elements, element_type

    raise ValueError(
        "Mesh data not available. Either pass model=model when calling plot(), "
        "or ensure results were saved with model data (results.save(..., model=model))."
    )


def _get_field_data(
    results: "Results",
    field: FieldName,
) -> tuple[np.ndarray, str, bool]:
    """Get field data array and metadata.

    Args:
        results: Results object.
        field: Field name to extract.

    Returns:
        Tuple of (data_array, display_name, is_cell_data).
        is_cell_data is True for element-centered data, False for nodal data.

    Raises:
        ValueError: If field name is unknown.
    """
    # Nodal fields
    if field == "displacement":
        data = results.displacement()
        return data, "Displacement", False

    if field == "displacement_magnitude":
        data = results.displacement_magnitude()
        return data, "Displacement Magnitude", False

    if field == "displacement_x":
        data = results.displacement()[:, 0]
        return data, "Displacement X", False

    if field == "displacement_y":
        data = results.displacement()[:, 1]
        return data, "Displacement Y", False

    if field == "displacement_z":
        data = results.displacement()[:, 2]
        return data, "Displacement Z", False

    # Element (cell) fields
    if field == "von_mises":
        data = results.von_mises()
        return data, "Von Mises Stress", True

    if field == "stress_xx":
        data = results.stress()[:, 0]
        return data, "Stress XX", True

    if field == "stress_yy":
        data = results.stress()[:, 1]
        return data, "Stress YY", True

    if field == "stress_zz":
        data = results.stress()[:, 2]
        return data, "Stress ZZ", True

    if field == "stress_xy":
        data = results.stress()[:, 3]
        return data, "Stress XY", True

    if field == "stress_yz":
        data = results.stress()[:, 4]
        return data, "Stress YZ", True

    if field == "stress_xz":
        data = results.stress()[:, 5]
        return data, "Stress XZ", True

    if field == "principal_1":
        data = results.principal_stresses()[:, 0]
        return data, "Principal Stress 1", True

    if field == "principal_2":
        data = results.principal_stresses()[:, 1]
        return data, "Principal Stress 2", True

    if field == "principal_3":
        data = results.principal_stresses()[:, 2]
        return data, "Principal Stress 3", True

    if field == "tresca":
        data = results.tresca()
        return data, "Tresca Stress", True

    if field == "hydrostatic":
        data = results.hydrostatic_stress()
        return data, "Hydrostatic Stress", True

    raise ValueError(
        f"Unknown field: {field}. Available fields: "
        f"displacement, displacement_magnitude, displacement_x/y/z, "
        f"von_mises, stress_xx/yy/zz/xy/yz/xz, principal_1/2/3, tresca, hydrostatic"
    )


def create_pyvista_mesh(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_type: str,
) -> "pyvista.UnstructuredGrid":
    """Create a PyVista UnstructuredGrid from mesh data.

    Args:
        nodes: Nx3 array of node coordinates.
        elements: MxK array of element connectivity.
        element_type: Element type string (e.g., "tet4", "hex8").

    Returns:
        PyVista UnstructuredGrid mesh.

    Raises:
        ValueError: If element type is not supported.
    """
    import pyvista as pv

    if element_type not in VTK_CELL_TYPES:
        raise ValueError(
            f"Unsupported element type for visualization: {element_type}. "
            f"Supported: {list(VTK_CELL_TYPES.keys())}"
        )

    vtk_type = VTK_CELL_TYPES[element_type]
    n_elements = elements.shape[0]
    n_nodes_per_elem = elements.shape[1]

    # Build cells array for PyVista: [n_nodes, node0, node1, ..., n_nodes, ...]
    cells = np.empty((n_elements, n_nodes_per_elem + 1), dtype=np.int64)
    cells[:, 0] = n_nodes_per_elem
    cells[:, 1:] = elements
    cells = cells.ravel()

    # Cell types array
    cell_types = np.full(n_elements, vtk_type, dtype=np.uint8)

    # Create the mesh
    mesh = pv.UnstructuredGrid(cells, cell_types, nodes)

    return mesh


def plot_results(
    results: "Results",
    model: "Model | None" = None,
    field: FieldName = "von_mises",
    *,
    cmap: str | None = None,
    show_edges: bool = True,
    show_scalar_bar: bool = True,
    scalar_bar_title: str | None = None,
    clim: tuple[float, float] | None = None,
    deformed: bool = False,
    scale_factor: float = 1.0,
    opacity: float = 1.0,
    window_size: tuple[int, int] = (1024, 768),
    background: str = "white",
    title: str | None = None,
    screenshot: str | Path | None = None,
    notebook: bool | None = None,
    off_screen: bool | None = None,
    return_plotter: bool = False,
) -> "pyvista.Plotter | None":
    """Plot FEA results using PyVista.

    This function creates an interactive 3D visualization of the analysis results.
    In Jupyter notebooks, it renders inline. For scripts, it opens an interactive
    window.

    Args:
        results: Results object from solve().
        model: Model object (required for in-memory results, optional if results
            were loaded from HDF5 with mesh data).
        field: Field to visualize. Options:
            - "displacement", "displacement_magnitude", "displacement_x/y/z"
            - "von_mises", "stress_xx/yy/zz/xy/yz/xz"
            - "principal_1/2/3", "tresca", "hydrostatic"
        cmap: Colormap name (e.g., "jet", "viridis"). Uses field-appropriate
            default if not specified.
        show_edges: Show mesh edges (default True).
        show_scalar_bar: Show color bar (default True).
        scalar_bar_title: Custom title for scalar bar.
        clim: Color limits as (min, max). Auto-scaled if not specified.
        deformed: Show deformed shape (default False).
        scale_factor: Deformation scale factor (default 1.0).
        opacity: Surface opacity 0-1 (default 1.0).
        window_size: Window size in pixels (default (1024, 768)).
        background: Background color (default "white").
        title: Plot title.
        screenshot: Save screenshot to this path.
        notebook: Force notebook mode (auto-detected if None).
        off_screen: Render off-screen without display.
        return_plotter: Return the plotter object instead of showing.

    Returns:
        PyVista Plotter object if return_plotter=True, else None.

    Example::

        # Basic usage
        results.plot("von_mises")

        # Customized
        results.plot(
            "displacement_magnitude",
            cmap="plasma",
            deformed=True,
            scale_factor=100,
            title="Cantilever Beam Displacement",
        )

        # Save to file
        results.plot("von_mises", screenshot="stress_plot.png")
    """
    _check_pyvista()
    import pyvista as pv

    # Get mesh data
    nodes, elements, element_type = _get_mesh_from_model_or_results(results, model)

    # Get field data
    field_data, field_name, is_cell_data = _get_field_data(results, field)

    # Apply deformation if requested
    if deformed:
        disp = results.displacement()
        nodes = nodes + disp * scale_factor

    # Create PyVista mesh
    pv_mesh = create_pyvista_mesh(nodes, elements, element_type)

    # Add field data
    if is_cell_data:
        pv_mesh.cell_data[field_name] = field_data
    else:
        pv_mesh.point_data[field_name] = field_data

    # Set active scalars
    if is_cell_data:
        pv_mesh.set_active_scalars(field_name, preference="cell")
    else:
        pv_mesh.set_active_scalars(field_name, preference="point")

    # Determine colormap
    if cmap is None:
        # Use field-appropriate default
        for key, default_cmap in DEFAULT_COLORMAPS.items():
            if key in field.lower():
                cmap = default_cmap
                break
        else:
            cmap = "viridis"

    # Determine rendering mode
    if notebook is None:
        notebook = _is_notebook()

    if off_screen is None:
        off_screen = screenshot is not None and not notebook

    # Create plotter
    plotter = pv.Plotter(
        window_size=window_size,
        notebook=notebook,
        off_screen=off_screen,
    )

    # Set background
    plotter.set_background(background)

    # Add mesh
    plotter.add_mesh(
        pv_mesh,
        scalars=field_name,
        cmap=cmap,
        show_edges=show_edges,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={
            "title": scalar_bar_title or field_name,
            "vertical": True,
        },
        clim=clim,
        opacity=opacity,
    )

    # Add title
    if title:
        plotter.add_title(title, font_size=12)

    # Add axes
    plotter.add_axes()

    # Take screenshot if requested
    if screenshot:
        plotter.screenshot(str(screenshot))

    # Return or show
    if return_plotter:
        return plotter

    if not off_screen:
        plotter.show()

    return None


def export_vtu(
    results: "Results",
    path: str | Path,
    model: "Model | None" = None,
    *,
    deformed: bool = False,
    scale_factor: float = 1.0,
    fields: list[FieldName] | None = None,
) -> None:
    """Export results to VTU format for ParaView.

    Creates a VTK Unstructured Grid file (.vtu) that can be opened in ParaView
    or other VTK-compatible visualization tools.

    Args:
        results: Results object from solve().
        path: Output file path (should end in .vtu).
        model: Model object (required for in-memory results).
        deformed: Export deformed shape (default False).
        scale_factor: Deformation scale factor (default 1.0).
        fields: List of fields to include. If None, exports all available fields.

    Example::

        # Export with all fields
        results.export("output.vtu", model=model)

        # Export specific fields only
        results.export(
            "output.vtu",
            model=model,
            fields=["displacement", "von_mises"],
        )

        # Export deformed shape
        results.export(
            "deformed.vtu",
            model=model,
            deformed=True,
            scale_factor=100,
        )
    """
    _check_pyvista()

    path = Path(path)
    if path.suffix.lower() != ".vtu":
        path = path.with_suffix(".vtu")

    # Get mesh data
    nodes, elements, element_type = _get_mesh_from_model_or_results(results, model)

    # Apply deformation if requested
    if deformed:
        disp = results.displacement()
        nodes = nodes + disp * scale_factor

    # Create PyVista mesh
    pv_mesh = create_pyvista_mesh(nodes, elements, element_type)

    # Determine which fields to export
    if fields is None:
        fields = [
            "displacement",
            "displacement_magnitude",
            "von_mises",
            "principal_1",
            "principal_2",
            "principal_3",
            "tresca",
            "hydrostatic",
        ]

    # Add all requested fields
    for field in fields:
        try:
            field_data, field_name, is_cell_data = _get_field_data(results, field)
            if is_cell_data:
                pv_mesh.cell_data[field_name] = field_data
            else:
                pv_mesh.point_data[field_name] = field_data
        except Exception:
            # Skip fields that can't be computed
            pass

    # Save to file
    pv_mesh.save(str(path))


def _is_notebook() -> bool:
    """Detect if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return False
        shell = ipython.__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal IPython
        else:
            return False
    except (ImportError, NameError, AttributeError):
        return False


def plot_mesh(
    model: "Model",
    *,
    show_edges: bool = True,
    show_nodes: bool = False,
    node_size: float = 5.0,
    opacity: float = 1.0,
    color: str = "lightgray",
    edge_color: str = "black",
    window_size: tuple[int, int] = (1024, 768),
    background: str = "white",
    title: str | None = None,
    screenshot: str | Path | None = None,
    notebook: bool | None = None,
    off_screen: bool | None = None,
    return_plotter: bool = False,
) -> "pyvista.Plotter | None":
    """Plot mesh geometry using PyVista.

    Useful for visualizing the mesh before solving.

    Args:
        model: Model with mesh to visualize.
        show_edges: Show mesh edges (default True).
        show_nodes: Show node markers (default False).
        node_size: Node marker size (default 5.0).
        opacity: Surface opacity (default 1.0).
        color: Surface color (default "lightgray").
        edge_color: Edge color (default "black").
        window_size: Window size in pixels.
        background: Background color.
        title: Plot title.
        screenshot: Save to this path.
        notebook: Force notebook mode.
        off_screen: Render off-screen without display.
        return_plotter: Return plotter instead of showing.

    Returns:
        PyVista Plotter if return_plotter=True, else None.
    """
    _check_pyvista()
    import pyvista as pv

    mesh = model.mesh
    nodes = mesh.coords
    elements = mesh.elements
    element_type = mesh.element_type

    # Create PyVista mesh
    pv_mesh = create_pyvista_mesh(nodes, elements, element_type)

    # Determine rendering mode
    if notebook is None:
        notebook = _is_notebook()

    if off_screen is None:
        off_screen = screenshot is not None and not notebook

    # Create plotter
    plotter = pv.Plotter(
        window_size=window_size,
        notebook=notebook,
        off_screen=off_screen,
    )

    plotter.set_background(background)

    # Add mesh surface
    plotter.add_mesh(
        pv_mesh,
        color=color,
        show_edges=show_edges,
        edge_color=edge_color,
        opacity=opacity,
    )

    # Add nodes if requested
    if show_nodes:
        plotter.add_points(
            nodes,
            color="blue",
            point_size=node_size,
            render_points_as_spheres=True,
        )

    # Add title
    if title:
        plotter.add_title(title, font_size=12)

    # Add axes
    plotter.add_axes()

    # Take screenshot
    if screenshot:
        plotter.screenshot(str(screenshot))

    if return_plotter:
        return plotter

    if not off_screen:
        plotter.show()

    return None
