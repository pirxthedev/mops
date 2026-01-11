"""MOPS - Modular Open Physics Solver.

A high-performance finite element analysis library with a declarative Python API.

Example usage::

    import mops
    from mops import Model, Mesh, Material, Nodes, Elements
    from mops import Force, solve

    # Create mesh from arrays
    mesh = Mesh(nodes, elements, element_type="tet4")

    # Define material
    steel = Material.steel()

    # Build model with immutable copy-on-write semantics
    model = (
        Model(mesh, materials={"steel": steel})
        .assign(Elements.all(), material="steel")
        .constrain(Nodes.where(x=0), dofs=["ux", "uy", "uz"])
        .load(Nodes.where(x=100), Force(fy=-1000))
    )

    # Solve
    results = solve(model)
    print(f"Max displacement: {results.max_displacement():.3e} m")
"""

from mops._core import (
    Material,
    Mesh as _CoreMesh,
    Results as _CoreResults,
    SolverConfig,
    compute_element_stress,
    element_stiffness,
    element_volume,
    solve_simple as _solve_simple,
    solver_info,
    version,
)
from mops.mesh import Mesh, MeshError
from mops.model import Model
from mops.loads import Force, Pressure, Moment
from mops.query import Nodes, Elements, Faces
from mops.results import Results

# Visualization (optional - requires pyvista)
try:
    from mops.viz import plot_results, export_vtu, plot_mesh, create_pyvista_mesh
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False
    plot_results = None
    export_vtu = None
    plot_mesh = None
    create_pyvista_mesh = None


def _unwrap_mesh(mesh):
    """Unwrap Python Mesh wrapper to get the underlying Rust mesh."""
    if hasattr(mesh, "_inner"):
        return mesh._inner
    return mesh


def solve_simple(mesh, material, constrained_nodes, loaded_nodes, load_vector, config=None):
    """Solve a simple FEA problem.

    This is a convenience function for testing the solver pipeline.

    Args:
        mesh: Mesh object (either Python Mesh wrapper or core Mesh).
        material: Material properties.
        constrained_nodes: Array of node indices to constrain.
        loaded_nodes: Array of node indices to load.
        load_vector: 3D force vector [fx, fy, fz].
        config: Optional solver configuration.

    Returns:
        Results object with displacements, stresses, etc.
    """
    return _solve_simple(
        _unwrap_mesh(mesh),
        material,
        constrained_nodes,
        loaded_nodes,
        load_vector,
        config,
    )

__all__ = [
    # Core types
    "Material",
    "Mesh",
    "MeshError",
    "Results",
    "SolverConfig",
    # Python classes
    "Model",
    "Force",
    "Pressure",
    "Moment",
    "Nodes",
    "Elements",
    "Faces",
    # Functions
    "solve",
    "solve_simple",
    "element_stiffness",
    "element_volume",
    "compute_element_stress",
    "solver_info",
    "version",
    # Visualization (optional)
    "plot_results",
    "export_vtu",
    "plot_mesh",
    "create_pyvista_mesh",
]

__version__ = version()


def solve(model: Model, config: SolverConfig | None = None) -> Results:
    """Solve the FEA problem.

    This is a pure function - it does not modify the model.

    Args:
        model: Complete model with mesh, materials, constraints, loads
        config: Optional solver configuration

    Returns:
        Results object with displacements, stresses, etc.

    Raises:
        MopsError: If the model is incomplete or solver fails
    """
    core_results = model._solve(config)
    return Results(core_results)
