#!/usr/bin/env python3
"""Gate Support Column Wind Load Analysis - Texas Hill Country.

This example sizes a support column for an 18 ft x 5 ft gate under wind load.

Wind Load Calculation (ASCE 7-22):
- Texas Hill Country: Risk Category II, Exposure Category C (open terrain)
- Basic wind speed V = 115 mph (3-second gust, per ASCE 7-22 Figure 26.5-1B)
- Velocity pressure: qz = 0.00256 * Kz * Kzt * Kd * Ke * V^2
- Wind pressure on gate: p = qz * G * Cf

The column is modeled as a cantilever fixed at ground level.
Wind load from the gate creates both shear and moment at the base.

Run with:
    cd mops && pip install -e ./mops-python
    python examples/gate_column_wind_analysis.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from mops import (
    Elements,
    Force,
    Material,
    Mesh,
    Model,
    Nodes,
    solve,
)


# =============================================================================
# Wind Load Calculations per ASCE 7-22
# =============================================================================

@dataclass
class WindLoadParams:
    """ASCE 7-22 Wind Load Parameters for Texas Hill Country."""
    V: float = 115.0      # Basic wind speed (mph) - 3-second gust
    exposure: str = "C"   # Exposure category (open hill country terrain)
    risk_cat: str = "II"  # Risk category (standard occupancy)

    # Coefficients
    Kd: float = 0.85      # Wind directionality factor (Table 26.6-1, solid freestanding walls)
    Kzt: float = 1.0      # Topographic factor (assume flat)
    Ke: float = 1.0       # Ground elevation factor (sea level to 1000 ft)
    G: float = 0.85       # Gust effect factor (rigid structure)
    Cf: float = 1.3       # Force coefficient for solid freestanding wall (Table 27.4-1)


def get_Kz(z_ft: float, exposure: str = "C") -> float:
    """Get velocity pressure exposure coefficient Kz (Table 26.10-1).

    Args:
        z_ft: Height above ground (feet)
        exposure: Exposure category (B, C, or D)

    Returns:
        Kz coefficient
    """
    # Exposure C parameters
    alpha = 9.5
    zg = 900  # gradient height (ft)

    z = max(15.0, min(z_ft, zg))  # Kz is constant below 15 ft
    Kz = 2.01 * (z / zg) ** (2 / alpha)
    return Kz


def calculate_wind_pressure(params: WindLoadParams, height_ft: float) -> float:
    """Calculate design wind pressure per ASCE 7-22.

    Args:
        params: Wind load parameters
        height_ft: Height above ground (feet)

    Returns:
        Wind pressure in psf (pounds per square foot)
    """
    Kz = get_Kz(height_ft, params.exposure)

    # Velocity pressure (Eq. 26.10-1)
    qz = 0.00256 * Kz * params.Kzt * params.Kd * params.Ke * params.V**2

    # Design wind pressure on solid sign/wall
    p = qz * params.G * params.Cf

    return p


# =============================================================================
# Gate and Column Geometry
# =============================================================================

@dataclass
class GateGeometry:
    """Gate dimensions and column parameters."""
    gate_width_ft: float = 18.0   # Gate width (horizontal span)
    gate_height_ft: float = 5.0   # Gate height
    clearance_ft: float = 0.5     # Ground clearance

    @property
    def gate_area_sqft(self) -> float:
        return self.gate_width_ft * self.gate_height_ft

    @property
    def centroid_height_ft(self) -> float:
        """Height of gate centroid above ground."""
        return self.clearance_ft + self.gate_height_ft / 2

    @property
    def column_height_ft(self) -> float:
        """Total column height (to top of gate)."""
        return self.clearance_ft + self.gate_height_ft


@dataclass
class ColumnSection:
    """Steel tube column section properties."""
    outer_diameter_in: float  # Outer diameter (inches)
    wall_thickness_in: float  # Wall thickness (inches)

    @property
    def inner_diameter_in(self) -> float:
        return self.outer_diameter_in - 2 * self.wall_thickness_in

    @property
    def area_sqin(self) -> float:
        """Cross-sectional area (in^2)."""
        Do, Di = self.outer_diameter_in, self.inner_diameter_in
        return np.pi / 4 * (Do**2 - Di**2)

    @property
    def moment_of_inertia_in4(self) -> float:
        """Second moment of area I (in^4)."""
        Do, Di = self.outer_diameter_in, self.inner_diameter_in
        return np.pi / 64 * (Do**4 - Di**4)

    @property
    def section_modulus_in3(self) -> float:
        """Elastic section modulus S = I/c (in^3)."""
        return self.moment_of_inertia_in4 / (self.outer_diameter_in / 2)

    @property
    def plastic_modulus_in3(self) -> float:
        """Plastic section modulus Z (in^3)."""
        Do, Di = self.outer_diameter_in, self.inner_diameter_in
        return (Do**3 - Di**3) / 6


def calculate_column_loads(gate: GateGeometry, wind_params: WindLoadParams) -> Tuple[float, float]:
    """Calculate wind loads on column.

    Returns:
        Tuple of (total_force_lbs, base_moment_lb_ft)
    """
    # Calculate wind pressure at gate centroid height
    p_psf = calculate_wind_pressure(wind_params, gate.centroid_height_ft)

    # Total wind force on gate
    F_lbs = p_psf * gate.gate_area_sqft

    # Assume gate is hinged on ONE column (worst case - all load to one column)
    # For a swinging gate, the hinge column takes full wind load

    # Base moment (force * moment arm)
    M_lb_ft = F_lbs * gate.centroid_height_ft

    return F_lbs, M_lb_ft


# =============================================================================
# Analytical Check (Before FEA)
# =============================================================================

def analytical_column_check(
    column: ColumnSection,
    gate: GateGeometry,
    wind_force_lbs: float,
    base_moment_lb_ft: float,
) -> dict:
    """Perform analytical stress and deflection check.

    Args:
        column: Column section properties
        gate: Gate geometry
        wind_force_lbs: Total wind force (lbs)
        base_moment_lb_ft: Base moment (lb-ft)

    Returns:
        Dictionary of analytical results
    """
    # Steel properties
    Fy = 36000  # Yield stress (psi) - A36 steel
    E = 29e6    # Elastic modulus (psi)

    # Convert moment to lb-in
    M_lb_in = base_moment_lb_ft * 12

    # Bending stress at base
    sigma_bending = M_lb_in / column.section_modulus_in3  # psi

    # Shear stress (approximate, V*Q/I*t)
    # For thin-walled tube, tau_max ≈ 2V/A
    tau_max = 2 * wind_force_lbs / column.area_sqin  # psi

    # Combined stress (von Mises approximation)
    sigma_vm = np.sqrt(sigma_bending**2 + 3 * tau_max**2)

    # Demand/capacity ratio
    dcr = sigma_vm / (0.6 * Fy)  # Using 0.6*Fy allowable for bending

    # Tip deflection (cantilever with end load)
    L_in = gate.column_height_ft * 12
    delta_in = wind_force_lbs * L_in**3 / (3 * E * column.moment_of_inertia_in4)

    # Deflection limit (typically L/60 to L/120 for sign posts)
    delta_limit = L_in / 60
    deflection_ratio = delta_in / delta_limit

    return {
        "bending_stress_psi": sigma_bending,
        "shear_stress_psi": tau_max,
        "von_mises_psi": sigma_vm,
        "dcr": dcr,
        "tip_deflection_in": delta_in,
        "deflection_limit_in": delta_limit,
        "deflection_ratio": deflection_ratio,
        "Fy_psi": Fy,
        "E_psi": E,
    }


# =============================================================================
# MOPS FEA Model
# =============================================================================

def generate_tube_column_mesh(
    outer_diam: float,
    wall_thick: float,
    height: float,
    n_circum: int = 12,
    n_radial: int = 2,
    n_height: int = 20,
) -> Mesh:
    """Generate a hex8 mesh for a cylindrical tube column.

    Args:
        outer_diam: Outer diameter
        wall_thick: Wall thickness
        height: Column height
        n_circum: Elements around circumference
        n_radial: Elements through wall thickness
        n_height: Elements along height

    Returns:
        Hex8 mesh for the tube column
    """
    inner_radius = (outer_diam - 2 * wall_thick) / 2
    outer_radius = outer_diam / 2

    # Generate nodes
    theta_vals = np.linspace(0, 2 * np.pi, n_circum + 1)[:-1]  # Exclude last (same as first)
    r_vals = np.linspace(inner_radius, outer_radius, n_radial + 1)
    z_vals = np.linspace(0, height, n_height + 1)

    nodes = []
    node_map = {}  # (i_theta, j_r, k_z) -> node index

    idx = 0
    for k, z in enumerate(z_vals):
        for j, r in enumerate(r_vals):
            for i, theta in enumerate(theta_vals):
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                nodes.append([x, y, z])
                node_map[(i, j, k)] = idx
                idx += 1

    nodes = np.array(nodes, dtype=np.float64)

    # Generate elements
    elements = []
    n_theta = len(theta_vals)

    for k in range(n_height):
        for j in range(n_radial):
            for i in range(n_circum):
                # Handle wraparound in theta direction
                i_next = (i + 1) % n_theta

                # Hex8 connectivity
                n0 = node_map[(i, j, k)]
                n1 = node_map[(i_next, j, k)]
                n2 = node_map[(i_next, j + 1, k)]
                n3 = node_map[(i, j + 1, k)]
                n4 = node_map[(i, j, k + 1)]
                n5 = node_map[(i_next, j, k + 1)]
                n6 = node_map[(i_next, j + 1, k + 1)]
                n7 = node_map[(i, j + 1, k + 1)]

                elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

    elements = np.array(elements, dtype=np.int64)
    return Mesh(nodes, elements, "hex8")


def run_fea_analysis(
    column: ColumnSection,
    gate: GateGeometry,
    wind_force_lbs: float,
) -> dict:
    """Run MOPS FEA analysis on column.

    Args:
        column: Column section properties
        gate: Gate geometry
        wind_force_lbs: Total wind force (lbs)

    Returns:
        Dictionary of FEA results
    """
    # Convert to consistent units (inches, lbs -> psi)
    height_in = gate.column_height_ft * 12

    print(f"\nGenerating tube mesh (OD={column.outer_diameter_in}\", t={column.wall_thickness_in}\")...")
    mesh = generate_tube_column_mesh(
        outer_diam=column.outer_diameter_in,
        wall_thick=column.wall_thickness_in,
        height=height_in,
        n_circum=16,
        n_radial=2,
        n_height=24,
    )
    print(f"  Nodes: {mesh.n_nodes}")
    print(f"  Elements: {mesh.n_elements}")

    # Steel material (A36)
    E_psi = 29e6
    nu = 0.3
    steel = Material("steel", e=E_psi, nu=nu)

    # Count top nodes for load distribution
    z_max = height_in
    top_nodes = Nodes.where(z=z_max).evaluate(mesh)
    n_top = len(top_nodes)

    # Apply wind force distributed across top nodes (x-direction)
    force_per_node = wind_force_lbs / n_top

    print(f"  Top nodes: {n_top}")
    print(f"  Force per node: {force_per_node:.2f} lbs")

    # Build model
    model = (
        Model(mesh, materials={"steel": steel})
        .assign(Elements.all(), material="steel")
        .constrain(Nodes.where(z=0), dofs=["ux", "uy", "uz"])
        .load(Nodes.where(z=z_max), Force(fx=force_per_node))
    )

    print("  Solving...")
    results = solve(model)
    print("  Complete!")

    # Extract results
    disp = results.displacement()
    stress = results.stress()
    von_mises = results.von_mises()

    # Get max tip displacement
    top_x_disp = [disp[n, 0] for n in top_nodes]
    max_tip_disp = np.max(np.abs(top_x_disp))

    # Get max stress at base (z near 0)
    base_elements = []
    node_coords = mesh.nodes
    for elem_idx in range(mesh.n_elements):
        elem_nodes = mesh.connectivity[elem_idx]
        elem_z = np.mean([node_coords[n, 2] for n in elem_nodes])
        if elem_z < height_in * 0.1:  # Bottom 10%
            base_elements.append(elem_idx)

    if base_elements:
        base_von_mises = [von_mises[e] for e in base_elements]
        max_base_vm = np.max(base_von_mises)
        base_stress_xx = [stress[e, 0] for e in base_elements]
        max_base_stress = np.max(np.abs(base_stress_xx))
    else:
        max_base_vm = np.max(von_mises)
        max_base_stress = np.max(np.abs(stress[:, 0]))

    return {
        "max_displacement_in": results.max_displacement(),
        "tip_displacement_in": max_tip_disp,
        "max_von_mises_psi": np.max(von_mises),
        "base_von_mises_psi": max_base_vm,
        "base_bending_stress_psi": max_base_stress,
    }


# =============================================================================
# Column Sizing Study
# =============================================================================

# Standard Schedule 40 pipe sizes (approximate)
STANDARD_PIPES = [
    ColumnSection(2.375, 0.154),   # 2" Sch 40
    ColumnSection(2.875, 0.203),   # 2.5" Sch 40
    ColumnSection(3.5, 0.216),     # 3" Sch 40
    ColumnSection(4.0, 0.226),     # 3.5" Sch 40
    ColumnSection(4.5, 0.237),     # 4" Sch 40
    ColumnSection(5.563, 0.258),   # 5" Sch 40
    ColumnSection(6.625, 0.280),   # 6" Sch 40
]


def main():
    print("=" * 70)
    print("GATE SUPPORT COLUMN WIND LOAD ANALYSIS")
    print("Texas Hill Country - ASCE 7-22")
    print("=" * 70)

    # Define problem
    gate = GateGeometry(
        gate_width_ft=18.0,
        gate_height_ft=5.0,
        clearance_ft=0.5,
    )

    wind = WindLoadParams(
        V=115.0,        # mph - Texas Hill Country per ASCE 7-22
        exposure="C",   # Open terrain
    )

    print("\n--- GATE SPECIFICATIONS ---")
    print(f"Gate size: {gate.gate_width_ft:.1f} ft x {gate.gate_height_ft:.1f} ft")
    print(f"Gate area: {gate.gate_area_sqft:.1f} sq ft")
    print(f"Ground clearance: {gate.clearance_ft:.1f} ft")
    print(f"Column height: {gate.column_height_ft:.1f} ft")
    print(f"Gate centroid height: {gate.centroid_height_ft:.2f} ft")

    print("\n--- WIND LOAD PARAMETERS (ASCE 7-22) ---")
    print(f"Basic wind speed V: {wind.V:.0f} mph (3-sec gust)")
    print(f"Exposure category: {wind.exposure}")
    print(f"Risk category: {wind.risk_cat}")
    print(f"Kz at gate height: {get_Kz(gate.centroid_height_ft):.3f}")
    print(f"Wind directionality Kd: {wind.Kd}")
    print(f"Gust effect factor G: {wind.G}")
    print(f"Force coefficient Cf: {wind.Cf}")

    # Calculate wind pressure and loads
    p_psf = calculate_wind_pressure(wind, gate.centroid_height_ft)
    F_lbs, M_lb_ft = calculate_column_loads(gate, wind)

    print(f"\n--- CALCULATED WIND LOADS ---")
    print(f"Design wind pressure: {p_psf:.2f} psf")
    print(f"Total wind force on gate: {F_lbs:.0f} lbs ({F_lbs/gate.gate_width_ft:.0f} lbs/ft)")
    print(f"Base moment (single column, worst case): {M_lb_ft:.0f} lb-ft")
    print(f"                                         {M_lb_ft/1000:.2f} kip-ft")

    # Analytical sizing study
    print("\n" + "=" * 70)
    print("COLUMN SIZING STUDY (Analytical Check)")
    print("=" * 70)
    print(f"{'Size':<12} {'OD':<8} {'t':<8} {'I':<10} {'S':<10} {'Stress':<12} {'DCR':<8} {'Defl':<8} {'Status'}")
    print(f"{'':12} {'(in)':<8} {'(in)':<8} {'(in^4)':<10} {'(in^3)':<10} {'(psi)':<12} {'':<8} {'(in)':<8}")
    print("-" * 100)

    Fy = 36000  # A36 steel yield
    candidates = []

    for i, col in enumerate(STANDARD_PIPES):
        result = analytical_column_check(col, gate, F_lbs, M_lb_ft)

        ok_stress = result["dcr"] < 1.0
        ok_defl = result["deflection_ratio"] < 1.0
        status = "OK" if (ok_stress and ok_defl) else "NG"

        size_name = f"{col.outer_diameter_in:.3f}\" pipe"
        print(f"{size_name:<12} {col.outer_diameter_in:<8.3f} {col.wall_thickness_in:<8.3f} "
              f"{col.moment_of_inertia_in4:<10.3f} {col.section_modulus_in3:<10.3f} "
              f"{result['bending_stress_psi']:<12.0f} {result['dcr']:<8.2f} "
              f"{result['tip_deflection_in']:<8.2f} {status}")

        if ok_stress and ok_defl:
            candidates.append((col, result))

    print("-" * 100)
    print(f"Allowable stress: 0.6 * Fy = 0.6 * {Fy} = {0.6*Fy:.0f} psi")
    print(f"Deflection limit: L/60 = {gate.column_height_ft*12/60:.2f} in")

    if not candidates:
        print("\n*** WARNING: No standard pipe sizes adequate! Consider larger sections. ***")
        return

    # Select smallest adequate size
    selected_col, selected_result = candidates[0]

    print(f"\n--- SELECTED COLUMN ---")
    print(f"Pipe size: {selected_col.outer_diameter_in:.3f}\" OD x {selected_col.wall_thickness_in:.3f}\" wall")
    print(f"Section properties:")
    print(f"  Area: {selected_col.area_sqin:.3f} in^2")
    print(f"  Moment of inertia I: {selected_col.moment_of_inertia_in4:.3f} in^4")
    print(f"  Section modulus S: {selected_col.section_modulus_in3:.3f} in^3")

    print(f"\nAnalytical results:")
    print(f"  Bending stress: {selected_result['bending_stress_psi']:.0f} psi")
    print(f"  Shear stress: {selected_result['shear_stress_psi']:.0f} psi")
    print(f"  von Mises stress: {selected_result['von_mises_psi']:.0f} psi")
    print(f"  Demand/Capacity ratio: {selected_result['dcr']:.2f}")
    print(f"  Tip deflection: {selected_result['tip_deflection_in']:.2f} in")

    # Run FEA verification
    print("\n" + "=" * 70)
    print("FEA VERIFICATION (MOPS)")
    print("=" * 70)

    try:
        fea_results = run_fea_analysis(selected_col, gate, F_lbs)

        print(f"\nFEA Results:")
        print(f"  Max displacement: {fea_results['max_displacement_in']:.4f} in")
        print(f"  Tip displacement: {fea_results['tip_displacement_in']:.4f} in")
        print(f"  Max von Mises stress: {fea_results['max_von_mises_psi']:.0f} psi")
        print(f"  Base von Mises stress: {fea_results['base_von_mises_psi']:.0f} psi")

        print(f"\nComparison (Analytical vs FEA):")
        print(f"  Tip deflection: {selected_result['tip_deflection_in']:.3f} in (analytical) vs "
              f"{fea_results['tip_displacement_in']:.3f} in (FEA)")
        print(f"  Base stress: {selected_result['von_mises_psi']:.0f} psi (analytical) vs "
              f"{fea_results['base_von_mises_psi']:.0f} psi (FEA)")

    except Exception as e:
        print(f"\nFEA analysis failed: {e}")
        print("Proceeding with analytical results only.")

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"""
For an 18 ft x 5 ft gate in Texas Hill Country:

  COLUMN: {selected_col.outer_diameter_in:.0f}" Schedule 40 Steel Pipe
          (OD = {selected_col.outer_diameter_in:.3f}", wall = {selected_col.wall_thickness_in:.3f}")

  EMBEDMENT: Minimum 3 ft below grade in concrete footing
             Footing size: 18" x 18" x 36" deep (minimum)

  MATERIAL: ASTM A36 or A500 Grade B steel

  SAFETY FACTORS:
    - Stress utilization: {selected_result['dcr']*100:.0f}% of allowable
    - Deflection: {selected_result['tip_deflection_in']:.1f}" ({selected_result['deflection_ratio']*100:.0f}% of L/60 limit)

  NOTES:
    1. Wind load calculated per ASCE 7-22, V = 115 mph
    2. Assumes single hinge column takes full wind load (conservative)
    3. Consider galvanizing or painting for corrosion protection
    4. Gate hardware (hinges, latches) must be sized accordingly
    5. Consult local building department for permit requirements
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
