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
    Mesh,
    Results,
    SolverConfig,
    solve_simple,
    solver_info,
    version,
)
from mops.model import Model
from mops.loads import Force, Pressure, Moment
from mops.query import Nodes, Elements, Faces

__all__ = [
    # Core types from Rust
    "Material",
    "Mesh",
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
    "solver_info",
    "version",
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
    return model._solve(config)
