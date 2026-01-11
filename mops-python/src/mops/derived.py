"""Derived quantity computation for FEA results.

This module provides functions for computing derived quantities from stress and strain
tensors, such as principal stresses, von Mises stress, Tresca stress, etc.

These functions operate on numpy arrays of stress/strain data in Voigt notation:
[sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz] for stress
[eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz] for strain

All functions are vectorized and work on arrays of shape (n, 6) where n is the
number of elements or nodes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def von_mises_stress(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute von Mises equivalent stress.

    The von Mises stress is defined as:
    σ_vm = √[(σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²]/√2

    Or equivalently in Voigt notation:
    σ_vm = √[½((σ_xx-σ_yy)² + (σ_yy-σ_zz)² + (σ_zz-σ_xx)²) + 3(τ_xy² + τ_yz² + τ_xz²)]

    Args:
        stress: Array of shape (n, 6) containing stress tensors in Voigt notation.

    Returns:
        Array of shape (n,) containing von Mises stress values.
    """
    if stress.ndim == 1:
        stress = stress.reshape(1, -1)

    sxx = stress[:, 0]
    syy = stress[:, 1]
    szz = stress[:, 2]
    txy = stress[:, 3]
    tyz = stress[:, 4]
    txz = stress[:, 5]

    term1 = (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
    term2 = 6.0 * (txy**2 + tyz**2 + txz**2)

    return np.sqrt((term1 + term2) / 2.0)


def principal_stresses(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute principal stresses from stress tensors.

    Principal stresses are the eigenvalues of the stress tensor matrix,
    returned in descending order (σ₁ ≥ σ₂ ≥ σ₃).

    Args:
        stress: Array of shape (n, 6) containing stress tensors in Voigt notation.

    Returns:
        Array of shape (n, 3) containing principal stresses [σ₁, σ₂, σ₃] per element.
    """
    if stress.ndim == 1:
        stress = stress.reshape(1, -1)

    n = stress.shape[0]
    principals = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        s = stress[i]
        # Build 3x3 symmetric stress matrix
        matrix = np.array(
            [
                [s[0], s[3], s[5]],
                [s[3], s[1], s[4]],
                [s[5], s[4], s[2]],
            ]
        )
        eigenvalues = np.linalg.eigvalsh(matrix)
        # Sort in descending order
        principals[i] = np.sort(eigenvalues)[::-1]

    return principals


def tresca_stress(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Tresca (maximum shear) equivalent stress.

    The Tresca stress is defined as:
    σ_tresca = σ₁ - σ₃ = max principal - min principal

    This is twice the maximum shear stress and is used in maximum shear
    stress failure theory.

    Args:
        stress: Array of shape (n, 6) containing stress tensors in Voigt notation.

    Returns:
        Array of shape (n,) containing Tresca stress values.
    """
    principals = principal_stresses(stress)
    return principals[:, 0] - principals[:, 2]


def max_shear_stress(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute maximum shear stress.

    τ_max = (σ₁ - σ₃) / 2

    Args:
        stress: Array of shape (n, 6) containing stress tensors in Voigt notation.

    Returns:
        Array of shape (n,) containing maximum shear stress values.
    """
    return tresca_stress(stress) / 2.0


def hydrostatic_stress(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute hydrostatic (mean) stress.

    σ_h = (σ_xx + σ_yy + σ_zz) / 3

    Args:
        stress: Array of shape (n, 6) containing stress tensors in Voigt notation.

    Returns:
        Array of shape (n,) containing hydrostatic stress values.
    """
    if stress.ndim == 1:
        stress = stress.reshape(1, -1)
    return (stress[:, 0] + stress[:, 1] + stress[:, 2]) / 3.0


def pressure(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute pressure (negative hydrostatic stress).

    p = -σ_h = -(σ_xx + σ_yy + σ_zz) / 3

    Args:
        stress: Array of shape (n, 6) containing stress tensors in Voigt notation.

    Returns:
        Array of shape (n,) containing pressure values.
    """
    return -hydrostatic_stress(stress)


def stress_intensity(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute stress intensity (maximum principal stress difference).

    σ_int = max(|σ₁ - σ₂|, |σ₂ - σ₃|, |σ₃ - σ₁|)

    This is used in some codes as an alternative to Tresca stress.

    Args:
        stress: Array of shape (n, 6) containing stress tensors in Voigt notation.

    Returns:
        Array of shape (n,) containing stress intensity values.
    """
    principals = principal_stresses(stress)
    s1, s2, s3 = principals[:, 0], principals[:, 1], principals[:, 2]
    return np.maximum(np.abs(s1 - s2), np.maximum(np.abs(s2 - s3), np.abs(s3 - s1)))


def deviatoric_stress(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute deviatoric stress tensor.

    s' = σ - σ_h * I

    where σ_h is the hydrostatic stress and I is the identity tensor.
    The deviatoric stress tensor represents the shape-changing part of stress.

    Args:
        stress: Array of shape (n, 6) containing stress tensors in Voigt notation.

    Returns:
        Array of shape (n, 6) containing deviatoric stress tensors.
    """
    if stress.ndim == 1:
        stress = stress.reshape(1, -1)

    p = hydrostatic_stress(stress)[:, np.newaxis]
    deviatoric = stress.copy()
    deviatoric[:, 0] -= p.flatten()
    deviatoric[:, 1] -= p.flatten()
    deviatoric[:, 2] -= p.flatten()
    return deviatoric


# Strain computations


def von_mises_strain(strain: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute von Mises equivalent strain.

    The equivalent strain is the work-conjugate of von Mises stress and is
    useful for plasticity analysis.

    ε_eq = √(2/3) * √(e':e') where e' is deviatoric strain

    With engineering shear strains (γ = 2ε for off-diagonal):
    ε_eq = √(2/3) * √[½((ε_xx-ε_yy)² + (ε_yy-ε_zz)² + (ε_zz-ε_xx)²) + ¾(γ_xy² + γ_yz² + γ_xz²)]

    Args:
        strain: Array of shape (n, 6) containing strain tensors in Voigt notation
                with engineering shear strains.

    Returns:
        Array of shape (n,) containing equivalent strain values.
    """
    if strain.ndim == 1:
        strain = strain.reshape(1, -1)

    exx = strain[:, 0]
    eyy = strain[:, 1]
    ezz = strain[:, 2]
    gxy = strain[:, 3]  # Engineering shear
    gyz = strain[:, 4]
    gxz = strain[:, 5]

    term1 = (exx - eyy) ** 2 + (eyy - ezz) ** 2 + (ezz - exx) ** 2
    term2 = 1.5 * (gxy**2 + gyz**2 + gxz**2)

    return np.sqrt((term1 + term2) / 2.0) * np.sqrt(2.0 / 3.0)


def principal_strains(strain: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute principal strains from strain tensors.

    Principal strains are the eigenvalues of the strain tensor matrix,
    returned in descending order (ε₁ ≥ ε₂ ≥ ε₃).

    Args:
        strain: Array of shape (n, 6) containing strain tensors in Voigt notation
                with engineering shear strains.

    Returns:
        Array of shape (n, 3) containing principal strains [ε₁, ε₂, ε₃] per element.
    """
    if strain.ndim == 1:
        strain = strain.reshape(1, -1)

    n = strain.shape[0]
    principals = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        e = strain[i]
        # Build 3x3 symmetric strain matrix (convert engineering shear to tensor shear)
        matrix = np.array(
            [
                [e[0], e[3] / 2, e[5] / 2],
                [e[3] / 2, e[1], e[4] / 2],
                [e[5] / 2, e[4] / 2, e[2]],
            ]
        )
        eigenvalues = np.linalg.eigvalsh(matrix)
        # Sort in descending order
        principals[i] = np.sort(eigenvalues)[::-1]

    return principals


def max_shear_strain(strain: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute maximum shear strain.

    γ_max = ε₁ - ε₃

    Args:
        strain: Array of shape (n, 6) containing strain tensors in Voigt notation.

    Returns:
        Array of shape (n,) containing maximum shear strain values.
    """
    principals = principal_strains(strain)
    return principals[:, 0] - principals[:, 2]


def volumetric_strain(strain: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute volumetric strain.

    ε_vol = ε_xx + ε_yy + ε_zz

    This represents the change in volume per unit volume.

    Args:
        strain: Array of shape (n, 6) containing strain tensors in Voigt notation.

    Returns:
        Array of shape (n,) containing volumetric strain values.
    """
    if strain.ndim == 1:
        strain = strain.reshape(1, -1)
    return strain[:, 0] + strain[:, 1] + strain[:, 2]


def deviatoric_strain(strain: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute deviatoric strain tensor.

    e' = ε - ε_m * I where ε_m = ε_vol / 3

    Args:
        strain: Array of shape (n, 6) containing strain tensors in Voigt notation.

    Returns:
        Array of shape (n, 6) containing deviatoric strain tensors.
    """
    if strain.ndim == 1:
        strain = strain.reshape(1, -1)

    em = volumetric_strain(strain) / 3.0
    deviatoric = strain.copy()
    deviatoric[:, 0] -= em
    deviatoric[:, 1] -= em
    deviatoric[:, 2] -= em
    return deviatoric
