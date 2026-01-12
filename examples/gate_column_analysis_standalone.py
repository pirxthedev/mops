#!/usr/bin/env python3
"""Gate Support Column Wind Load Analysis - Texas Hill Country.

Standalone analytical version (no external dependencies).

Wind Load Calculation per ASCE 7-22:
- Texas Hill Country: Risk Category II, Exposure Category C (open terrain)
- Basic wind speed V = 115 mph (3-second gust, per ASCE 7-22 Figure 26.5-1B)

The column is modeled as a cantilever fixed at ground level.
"""

import math
from dataclasses import dataclass
from typing import Tuple, List

PI = math.pi

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
    Kd: float = 0.85      # Wind directionality factor (Table 26.6-1, solid signs)
    Kzt: float = 1.0      # Topographic factor (assume no speed-up)
    Ke: float = 1.0       # Ground elevation factor
    G: float = 0.85       # Gust effect factor (rigid structure)
    Cf: float = 1.3       # Force coefficient for solid freestanding wall


def get_Kz(z_ft: float, exposure: str = "C") -> float:
    """Get velocity pressure exposure coefficient Kz (Table 26.10-1)."""
    # Exposure C parameters
    alpha = 9.5
    zg = 900  # gradient height (ft)
    z = max(15.0, min(z_ft, zg))
    Kz = 2.01 * (z / zg) ** (2 / alpha)
    return Kz


def calculate_wind_pressure(params: WindLoadParams, height_ft: float) -> float:
    """Calculate design wind pressure per ASCE 7-22."""
    Kz = get_Kz(height_ft, params.exposure)
    qz = 0.00256 * Kz * params.Kzt * params.Kd * params.Ke * params.V**2
    p = qz * params.G * params.Cf
    return p


# =============================================================================
# Gate and Column Geometry
# =============================================================================

@dataclass
class GateGeometry:
    """Gate dimensions."""
    gate_width_ft: float = 18.0
    gate_height_ft: float = 5.0
    clearance_ft: float = 0.5

    @property
    def gate_area_sqft(self) -> float:
        return self.gate_width_ft * self.gate_height_ft

    @property
    def centroid_height_ft(self) -> float:
        return self.clearance_ft + self.gate_height_ft / 2

    @property
    def column_height_ft(self) -> float:
        return self.clearance_ft + self.gate_height_ft


@dataclass
class ColumnSection:
    """Steel tube column section."""
    name: str
    outer_diameter_in: float
    wall_thickness_in: float

    @property
    def inner_diameter_in(self) -> float:
        return self.outer_diameter_in - 2 * self.wall_thickness_in

    @property
    def area_sqin(self) -> float:
        Do, Di = self.outer_diameter_in, self.inner_diameter_in
        return PI / 4 * (Do**2 - Di**2)

    @property
    def moment_of_inertia_in4(self) -> float:
        Do, Di = self.outer_diameter_in, self.inner_diameter_in
        return PI / 64 * (Do**4 - Di**4)

    @property
    def section_modulus_in3(self) -> float:
        return self.moment_of_inertia_in4 / (self.outer_diameter_in / 2)

    @property
    def weight_per_ft(self) -> float:
        """Weight in lb/ft (steel = 490 lb/ft³)"""
        return self.area_sqin * 490 / 144


def calculate_column_loads(gate: GateGeometry, wind_params: WindLoadParams) -> Tuple[float, float]:
    """Calculate wind loads on column."""
    p_psf = calculate_wind_pressure(wind_params, gate.centroid_height_ft)
    F_lbs = p_psf * gate.gate_area_sqft
    M_lb_ft = F_lbs * gate.centroid_height_ft
    return F_lbs, M_lb_ft


def analytical_column_check(
    column: ColumnSection,
    gate: GateGeometry,
    wind_force_lbs: float,
    base_moment_lb_ft: float,
) -> dict:
    """Perform analytical stress and deflection check."""
    # Steel properties (A36)
    Fy = 36000  # psi
    E = 29e6    # psi

    M_lb_in = base_moment_lb_ft * 12
    sigma_bending = M_lb_in / column.section_modulus_in3
    tau_max = 2 * wind_force_lbs / column.area_sqin
    sigma_vm = math.sqrt(sigma_bending**2 + 3 * tau_max**2)
    dcr = sigma_vm / (0.6 * Fy)

    L_in = gate.column_height_ft * 12
    delta_in = wind_force_lbs * L_in**3 / (3 * E * column.moment_of_inertia_in4)
    delta_limit = L_in / 60

    return {
        "bending_stress_psi": sigma_bending,
        "shear_stress_psi": tau_max,
        "von_mises_psi": sigma_vm,
        "dcr": dcr,
        "tip_deflection_in": delta_in,
        "deflection_limit_in": delta_limit,
        "deflection_ratio": delta_in / delta_limit,
    }


# Standard pipe sizes (Schedule 40 and 80)
STANDARD_PIPES = [
    ColumnSection("2\" Sch 40", 2.375, 0.154),
    ColumnSection("2-1/2\" Sch 40", 2.875, 0.203),
    ColumnSection("3\" Sch 40", 3.500, 0.216),
    ColumnSection("3-1/2\" Sch 40", 4.000, 0.226),
    ColumnSection("4\" Sch 40", 4.500, 0.237),
    ColumnSection("5\" Sch 40", 5.563, 0.258),
    ColumnSection("6\" Sch 40", 6.625, 0.280),
    ColumnSection("4\" Sch 80", 4.500, 0.337),
    ColumnSection("5\" Sch 80", 5.563, 0.375),
    ColumnSection("6\" Sch 80", 6.625, 0.432),
]


def main():
    print("=" * 75)
    print("GATE SUPPORT COLUMN - WIND LOAD ANALYSIS")
    print("Location: Texas Hill Country | Code: ASCE 7-22")
    print("=" * 75)

    # Define problem
    gate = GateGeometry(gate_width_ft=18.0, gate_height_ft=5.0, clearance_ft=0.5)
    wind = WindLoadParams(V=115.0, exposure="C")

    print("\n+-----------------------------------------------------------------------+")
    print("| GATE SPECIFICATIONS                                                   |")
    print("+-----------------------------------------------------------------------+")
    print(f"| Gate size:           {gate.gate_width_ft:.0f} ft (W) x {gate.gate_height_ft:.0f} ft (H)                            |")
    print(f"| Gate area:           {gate.gate_area_sqft:.0f} sq ft                                           |")
    print(f"| Ground clearance:    {gate.clearance_ft:.1f} ft                                              |")
    print(f"| Column height:       {gate.column_height_ft:.1f} ft (above grade)                              |")
    print(f"| Centroid height:     {gate.centroid_height_ft:.1f} ft                                             |")
    print("+-----------------------------------------------------------------------+")

    print("\n+-----------------------------------------------------------------------+")
    print("| WIND LOAD PARAMETERS (ASCE 7-22)                                      |")
    print("+-----------------------------------------------------------------------+")
    print(f"| Basic wind speed:    V = {wind.V:.0f} mph (3-second gust at 33 ft)            |")
    print(f"| Exposure category:   {wind.exposure} (open terrain, scattered obstructions)         |")
    print(f"| Risk category:       {wind.risk_cat} (standard structures)                          |")
    print(f"| Kz (at {gate.centroid_height_ft:.1f} ft):        {get_Kz(gate.centroid_height_ft):.3f}                                            |")
    print(f"| Kd (directionality): {wind.Kd:.2f}                                              |")
    print(f"| G (gust factor):     {wind.G:.2f}                                              |")
    print(f"| Cf (force coeff):    {wind.Cf:.1f} (solid freestanding wall)                     |")
    print("+-----------------------------------------------------------------------+")

    # Calculate loads
    p_psf = calculate_wind_pressure(wind, gate.centroid_height_ft)
    F_lbs, M_lb_ft = calculate_column_loads(gate, wind)

    print("\n+-----------------------------------------------------------------------+")
    print("| DESIGN LOADS (Ultimate)                                               |")
    print("+-----------------------------------------------------------------------+")
    print(f"| Wind pressure:       p = {p_psf:.1f} psf                                     |")
    print(f"| Total wind force:    F = {F_lbs:.0f} lbs = {F_lbs/1000:.2f} kips                           |")
    print(f"| Base moment:         M = {M_lb_ft:.0f} lb-ft = {M_lb_ft/1000:.2f} kip-ft                   |")
    print(f"| Base shear:          V = {F_lbs:.0f} lbs                                        |")
    print("+-----------------------------------------------------------------------+")
    print("| Note: Assumes single hinge column takes full wind load (worst case)   |")
    print("+-----------------------------------------------------------------------+")

    # Sizing study
    print("\n" + "=" * 75)
    print("COLUMN SIZING STUDY - Standard Steel Pipe Sections")
    print("=" * 75)
    print(f"Material: ASTM A36 Steel (Fy = 36 ksi)")
    print(f"Allowable stress: Fb = 0.6 x Fy = 21.6 ksi")
    print(f"Deflection limit: L/60 = {gate.column_height_ft*12/60:.2f}\"")
    print()

    header = f"{'Section':<16} {'OD':<6} {'t':<6} {'I':<8} {'S':<7} {'Wt':<7} {'Stress':<8} {'DCR':<6} {'Defl':<6} {'Check'}"
    units =  f"{'':16} {'(in)':<6} {'(in)':<6} {'(in4)':<8} {'(in3)':<7} {'(lb/ft)':<7} {'(ksi)':<8} {'':<6} {'(in)':<6} {''}"
    print(header)
    print(units)
    print("-" * 85)

    candidates = []
    for col in STANDARD_PIPES:
        result = analytical_column_check(col, gate, F_lbs, M_lb_ft)

        ok_stress = result["dcr"] <= 1.0
        ok_defl = result["deflection_ratio"] <= 1.0

        if ok_stress and ok_defl:
            status = "[OK]"
            candidates.append((col, result))
        elif result["dcr"] > 1.5:
            status = "[FAIL]"
        else:
            status = "[NG]"

        print(f"{col.name:<16} {col.outer_diameter_in:<6.3f} {col.wall_thickness_in:<6.3f} "
              f"{col.moment_of_inertia_in4:<8.2f} {col.section_modulus_in3:<7.2f} "
              f"{col.weight_per_ft:<7.1f} {result['bending_stress_psi']/1000:<8.1f} "
              f"{result['dcr']:<6.2f} {result['tip_deflection_in']:<6.2f} {status}")

    print("-" * 85)

    if not candidates:
        print("\n*** No standard sections adequate. Consider custom steel tube. ***")
        return

    # Select smallest adequate
    selected_col, selected_result = candidates[0]

    print("\n" + "=" * 75)
    print("SELECTED COLUMN")
    print("=" * 75)
    print(f"""
    PIPE:   {selected_col.name}
            OD = {selected_col.outer_diameter_in:.3f}"
            Wall = {selected_col.wall_thickness_in:.3f}"
            Weight = {selected_col.weight_per_ft:.1f} lb/ft

    SECTION PROPERTIES:
            A  = {selected_col.area_sqin:.2f} in^2
            I  = {selected_col.moment_of_inertia_in4:.2f} in^4
            S  = {selected_col.section_modulus_in3:.2f} in^3

    STRESS CHECK:
            Stress_bending = {selected_result['bending_stress_psi']/1000:.1f} ksi
            Stress_allow   = 21.6 ksi
            DCR            = {selected_result['dcr']:.2f} ({selected_result['dcr']*100:.0f}% utilized)

    DEFLECTION CHECK:
            Defl_calc    = {selected_result['tip_deflection_in']:.2f}"
            Defl_allow   = {selected_result['deflection_limit_in']:.2f}" (L/60)
            Ratio        = {selected_result['deflection_ratio']:.2f} ({selected_result['deflection_ratio']*100:.0f}% of limit)
""")

    # Foundation recommendations
    embedment_ft = max(3.0, gate.column_height_ft * 0.4)
    footing_size = max(18, selected_col.outer_diameter_in * 3)

    print("=" * 75)
    print("FOUNDATION RECOMMENDATIONS")
    print("=" * 75)
    print(f"""
    CONCRETE FOOTING:
            Embedment depth:  {embedment_ft:.0f} ft minimum (below grade)
            Footing size:     {footing_size:.0f}" x {footing_size:.0f}" x {embedment_ft*12:.0f}" deep
            Concrete:         3000 psi minimum

    SOIL ASSUMPTIONS:
            Assumed bearing:  1500 psf (verify with local geotech)
            Texas Hill Country typically has limestone/caliche

    INSTALLATION:
            1. Excavate hole {footing_size + 6:.0f}" diameter x {embedment_ft*12 + 6:.0f}" deep
            2. Place 6" gravel base
            3. Set column plumb with bracing
            4. Pour concrete, vibrate to consolidate
            5. Cure minimum 7 days before loading
""")

    # Final spec sheet
    print("=" * 75)
    print("ENGINEERING SPECIFICATION SUMMARY")
    print("=" * 75)
    print(f"""
    PROJECT:        18' x 5' Swing Gate - Texas Hill Country
    WIND DESIGN:    ASCE 7-22, V = 115 mph, Exposure C

    COLUMN SPECIFICATION:
    +-- Material:       ASTM A500 Grade B or A36 Steel Pipe
    +-- Section:        {selected_col.name}
    +-- Length:         {gate.column_height_ft + embedment_ft:.0f} ft total ({gate.column_height_ft:.0f}' above + {embedment_ft:.0f}' embedded)
    +-- Finish:         Hot-dip galvanized per ASTM A123, or
    |                   Prime + 2 coats alkyd enamel (touch up welds)
    +-- Quantity:       2 (one hinge post, one latch post)

    FOUNDATION:
    +-- Type:           Concrete pier
    +-- Size:           {footing_size:.0f}" x {footing_size:.0f}" x {embedment_ft*12:.0f}" deep
    +-- Concrete:       f'c = 3000 psi minimum
    +-- Reinforcing:    (4) #4 vertical bars with #3 ties @ 12" o.c.

    DESIGN SUMMARY:
    +-- Wind force:     {F_lbs:.0f} lbs
    +-- Base moment:    {M_lb_ft/1000:.2f} kip-ft
    +-- Stress ratio:   {selected_result['dcr']*100:.0f}%
    +-- Deflection:     {selected_result['tip_deflection_in']:.1f}" < {selected_result['deflection_limit_in']:.1f}" (L/60)

    NOTES:
    1. Column designed for full wind load on single post (conservative)
    2. Verify soil bearing capacity with local geotechnical data
    3. Gate hardware (hinges, latches) to be sized for wind loads
    4. Consult local building department for permit requirements
    5. Licensed PE stamp may be required for permit
""")

    print("=" * 75)
    print("Analysis complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()
