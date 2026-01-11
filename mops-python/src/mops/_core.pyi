"""Type stubs for Rust bindings."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

class Material:
    """Material definition for Python."""

    def __init__(
        self, name: str, e: float, nu: float, rho: float = 0.0
    ) -> None: ...
    @staticmethod
    def steel() -> Material: ...
    @staticmethod
    def aluminum() -> Material: ...
    @property
    def name(self) -> str: ...
    @property
    def e(self) -> float: ...
    @property
    def nu(self) -> float: ...
    @property
    def rho(self) -> float: ...

class Mesh:
    """Mesh data for Python."""

    def __init__(
        self,
        nodes: NDArray[np.float64],
        elements: NDArray[np.int64],
        element_type: str,
    ) -> None: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def n_elements(self) -> int: ...

class SolverConfig:
    """Solver configuration."""

    def __init__(
        self,
        solver_type: Literal["auto", "direct", "iterative"] = "auto",
        auto_threshold: int = 100_000,
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
    ) -> None: ...

class Results:
    """FEA solution results."""

    def displacement(self) -> NDArray[np.float64]: ...
    def displacement_magnitude(self) -> NDArray[np.float64]: ...
    def max_displacement(self) -> float: ...

def solve_simple(
    mesh: Mesh,
    material: Material,
    constrained_nodes: NDArray[np.int64],
    loaded_nodes: NDArray[np.int64],
    load_vector: NDArray[np.float64],
    config: SolverConfig | None = None,
) -> Results: ...
def solve_with_forces(
    mesh: Mesh,
    material: Material,
    constraints: NDArray[np.float64],
    forces: NDArray[np.float64],
    config: SolverConfig | None = None,
) -> Results: ...
def solver_info() -> dict[str, bool]: ...
def version() -> str: ...
