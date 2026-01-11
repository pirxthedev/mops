"""Load type definitions."""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Force:
    """Concentrated force at nodes.

    Args:
        fx: Force component in X direction (N)
        fy: Force component in Y direction (N)
        fz: Force component in Z direction (N)

    Example::

        # Apply 1000 N downward force
        Force(fy=-1000)

        # Apply force in arbitrary direction
        Force(fx=100, fy=-200, fz=50)
    """

    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0

    def __post_init__(self):
        if self.fx == 0.0 and self.fy == 0.0 and self.fz == 0.0:
            raise ValueError("Force cannot be zero in all directions")


@dataclass(frozen=True)
class Pressure:
    """Surface pressure on faces.

    Args:
        value: Pressure magnitude (Pa), positive = into surface

    Example::

        # Apply 1 MPa pressure
        Pressure(1e6)
    """

    value: float

    def __post_init__(self):
        if self.value == 0.0:
            raise ValueError("Pressure cannot be zero")


@dataclass(frozen=True)
class Moment:
    """Concentrated moment at nodes.

    Args:
        mx: Moment about X axis (N·m)
        my: Moment about Y axis (N·m)
        mz: Moment about Z axis (N·m)

    Example::

        # Apply 100 N·m moment about Z axis
        Moment(mz=100)
    """

    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0

    def __post_init__(self):
        if self.mx == 0.0 and self.my == 0.0 and self.mz == 0.0:
            raise ValueError("Moment cannot be zero in all directions")


# Union type for all load types
Load = Union[Force, Pressure, Moment]
