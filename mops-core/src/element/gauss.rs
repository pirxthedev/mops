//! Gauss quadrature rules for numerical integration.
//!
//! This module provides standard Gauss-Legendre quadrature rules for:
//! - 1D line integration
//! - Tetrahedral volume integration
//! - Hexahedral (brick) volume integration
//!
//! # Usage
//!
//! ```
//! use mops_core::element::gauss::{gauss_1d, gauss_tet, gauss_hex};
//!
//! // 2-point 1D rule
//! for (xi, w) in gauss_1d(2) {
//!     // integrate at point xi with weight w
//! }
//!
//! // 4-point tetrahedral rule
//! for gp in gauss_tet(4) {
//!     // gp.coords gives (L1, L2, L3, L4) barycentric coordinates
//!     // gp.weight is the integration weight
//! }
//! ```

/// A Gauss quadrature point with natural coordinates and weight.
#[derive(Debug, Clone, Copy)]
pub struct GaussPoint {
    /// Natural coordinates.
    /// - For 1D: [ξ, 0, 0]
    /// - For tetrahedral: [L1, L2, L3, L4] (barycentric, sum=1, stored in coords[0..3] with L4 implicit)
    /// - For hexahedral: [ξ, η, ζ] in [-1, 1]³
    pub coords: [f64; 4],
    /// Integration weight.
    pub weight: f64,
}

impl GaussPoint {
    /// Create a new Gauss point.
    pub fn new(coords: [f64; 4], weight: f64) -> Self {
        Self { coords, weight }
    }

    /// Get ξ (first natural coordinate).
    #[inline]
    pub fn xi(&self) -> f64 {
        self.coords[0]
    }

    /// Get η (second natural coordinate).
    #[inline]
    pub fn eta(&self) -> f64 {
        self.coords[1]
    }

    /// Get ζ (third natural coordinate).
    #[inline]
    pub fn zeta(&self) -> f64 {
        self.coords[2]
    }
}

/// 1D Gauss-Legendre quadrature points and weights.
///
/// Returns (point, weight) pairs for integration on [-1, 1].
///
/// # Arguments
///
/// * `n` - Number of integration points (1, 2, 3, or 4)
///
/// # Panics
///
/// Panics if `n` is not in 1..=4.
pub fn gauss_1d(n: usize) -> Vec<(f64, f64)> {
    match n {
        1 => vec![(0.0, 2.0)],
        2 => {
            let p = 1.0 / 3.0_f64.sqrt();
            vec![(-p, 1.0), (p, 1.0)]
        }
        3 => {
            let p = (3.0 / 5.0_f64).sqrt();
            vec![(-p, 5.0 / 9.0), (0.0, 8.0 / 9.0), (p, 5.0 / 9.0)]
        }
        4 => {
            // Points: ±√((3 ∓ 2√(6/5))/7)
            let sqrt_6_5 = (6.0 / 5.0_f64).sqrt();
            let p1 = ((3.0 - 2.0 * sqrt_6_5) / 7.0).sqrt();
            let p2 = ((3.0 + 2.0 * sqrt_6_5) / 7.0).sqrt();
            // Weights: (18 ± √30) / 36
            let sqrt_30 = 30.0_f64.sqrt();
            let w1 = (18.0 + sqrt_30) / 36.0;
            let w2 = (18.0 - sqrt_30) / 36.0;
            vec![(-p2, w2), (-p1, w1), (p1, w1), (p2, w2)]
        }
        _ => panic!("gauss_1d: n must be 1, 2, 3, or 4, got {}", n),
    }
}

/// Tetrahedral Gauss quadrature points.
///
/// Returns integration points for a unit tetrahedron with vertices at:
/// - (0, 0, 0)
/// - (1, 0, 0)
/// - (0, 1, 0)
/// - (0, 0, 1)
///
/// Points are given in barycentric coordinates (L1, L2, L3, L4) where Li ≥ 0 and ΣLi = 1.
/// The coords array stores [L1, L2, L3, L4].
///
/// Weights are scaled for the unit tetrahedron (volume = 1/6), so ∫f dV ≈ Σ w_i * f(x_i).
///
/// # Arguments
///
/// * `n` - Number of integration points (1, 4, or 5)
///
/// # Integration Order
///
/// - n=1: Exact for polynomials up to degree 1 (linear)
/// - n=4: Exact for polynomials up to degree 2 (quadratic)
/// - n=5: Exact for polynomials up to degree 3 (cubic)
///
/// # Panics
///
/// Panics if `n` is not 1, 4, or 5.
pub fn gauss_tet(n: usize) -> Vec<GaussPoint> {
    match n {
        1 => {
            // 1-point rule: centroid
            // Weight = volume of unit tet = 1/6
            vec![GaussPoint::new(
                [0.25, 0.25, 0.25, 0.25],
                1.0 / 6.0,
            )]
        }
        4 => {
            // 4-point rule (degree 2)
            // Points at (α, β, β, β) and permutations
            // α = (5 + 3√5) / 20 ≈ 0.5854
            // β = (5 - √5) / 20 ≈ 0.1382
            let sqrt5 = 5.0_f64.sqrt();
            let alpha = (5.0 + 3.0 * sqrt5) / 20.0;
            let beta = (5.0 - sqrt5) / 20.0;
            // Weight = (1/6) / 4 = 1/24 for each point
            let w = 1.0 / 24.0;
            vec![
                GaussPoint::new([alpha, beta, beta, beta], w),
                GaussPoint::new([beta, alpha, beta, beta], w),
                GaussPoint::new([beta, beta, alpha, beta], w),
                GaussPoint::new([beta, beta, beta, alpha], w),
            ]
        }
        5 => {
            // 5-point rule (degree 3) - Keast rule
            // 1 point at centroid, 4 points at vertices
            // Centroid weight: -4/5 * (1/6) = -2/15
            // Vertex weight: 9/20 * (1/6) = 3/40
            // Sum check: -2/15 + 4 * 3/40 = -2/15 + 3/10 = -4/30 + 9/30 = 5/30 = 1/6 ✓
            let w_center = -2.0 / 15.0;
            let w_vertex = 3.0 / 40.0;
            vec![
                GaussPoint::new([0.25, 0.25, 0.25, 0.25], w_center),
                GaussPoint::new([1.0, 0.0, 0.0, 0.0], w_vertex),
                GaussPoint::new([0.0, 1.0, 0.0, 0.0], w_vertex),
                GaussPoint::new([0.0, 0.0, 1.0, 0.0], w_vertex),
                GaussPoint::new([0.0, 0.0, 0.0, 1.0], w_vertex),
            ]
        }
        _ => panic!("gauss_tet: n must be 1, 4, or 5, got {}", n),
    }
}

/// Hexahedral Gauss quadrature points.
///
/// Returns integration points for a reference hexahedron with ξ, η, ζ ∈ [-1, 1].
///
/// Uses tensor product of 1D Gauss-Legendre rules.
///
/// # Arguments
///
/// * `n` - Number of points per direction (1, 2, or 3)
///
/// Returns n³ total integration points.
///
/// # Integration Order
///
/// - n=1: Exact for polynomials up to degree 1
/// - n=2: Exact for polynomials up to degree 3 (standard for Hex8)
/// - n=3: Exact for polynomials up to degree 5 (for Hex20)
///
/// # Panics
///
/// Panics if `n` is not 1, 2, or 3.
pub fn gauss_hex(n: usize) -> Vec<GaussPoint> {
    if !(1..=3).contains(&n) {
        panic!("gauss_hex: n must be 1, 2, or 3, got {}", n);
    }

    let rule_1d = gauss_1d(n);
    let mut points = Vec::with_capacity(n * n * n);

    for &(xi, w_xi) in &rule_1d {
        for &(eta, w_eta) in &rule_1d {
            for &(zeta, w_zeta) in &rule_1d {
                points.push(GaussPoint::new(
                    [xi, eta, zeta, 0.0],
                    w_xi * w_eta * w_zeta,
                ));
            }
        }
    }

    points
}

/// Triangle Gauss quadrature points for 2D elements.
///
/// Returns integration points for a unit triangle with vertices at (0,0), (1,0), (0,1).
/// Points are given in area coordinates (L1, L2, L3) where Li ≥ 0 and ΣLi = 1.
/// The coords array stores [L1, L2, L3, 0].
///
/// Weights are scaled for the unit triangle (area = 1/2).
///
/// # Arguments
///
/// * `n` - Number of integration points (1, 3, or 4)
///
/// # Panics
///
/// Panics if `n` is not 1, 3, or 4.
pub fn gauss_tri(n: usize) -> Vec<GaussPoint> {
    match n {
        1 => {
            // 1-point rule: centroid, degree 1
            vec![GaussPoint::new([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], 0.5)]
        }
        3 => {
            // 3-point rule: edge midpoints, degree 2
            let w = 1.0 / 6.0;
            vec![
                GaussPoint::new([0.5, 0.5, 0.0, 0.0], w),
                GaussPoint::new([0.0, 0.5, 0.5, 0.0], w),
                GaussPoint::new([0.5, 0.0, 0.5, 0.0], w),
            ]
        }
        4 => {
            // 4-point rule: centroid + 3 points, degree 3
            let w_center = -27.0 / 96.0;
            let w_corner = 25.0 / 96.0;
            vec![
                GaussPoint::new([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], w_center),
                GaussPoint::new([0.6, 0.2, 0.2, 0.0], w_corner),
                GaussPoint::new([0.2, 0.6, 0.2, 0.0], w_corner),
                GaussPoint::new([0.2, 0.2, 0.6, 0.0], w_corner),
            ]
        }
        _ => panic!("gauss_tri: n must be 1, 3, or 4, got {}", n),
    }
}

/// Quadrilateral Gauss quadrature points for 2D elements.
///
/// Returns integration points for a reference quadrilateral with ξ, η ∈ [-1, 1].
///
/// Uses tensor product of 1D Gauss-Legendre rules.
///
/// # Arguments
///
/// * `n` - Number of points per direction (1, 2, or 3)
///
/// Returns n² total integration points.
pub fn gauss_quad(n: usize) -> Vec<GaussPoint> {
    if !(1..=3).contains(&n) {
        panic!("gauss_quad: n must be 1, 2, or 3, got {}", n);
    }

    let rule_1d = gauss_1d(n);
    let mut points = Vec::with_capacity(n * n);

    for &(xi, w_xi) in &rule_1d {
        for &(eta, w_eta) in &rule_1d {
            points.push(GaussPoint::new([xi, eta, 0.0, 0.0], w_xi * w_eta));
        }
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gauss_1d_integrates_polynomials() {
        // 1-point rule integrates constants exactly
        // ∫_{-1}^{1} 3 dx = 6
        let rule = gauss_1d(1);
        let integral: f64 = rule.iter().map(|&(_, w)| 3.0 * w).sum();
        assert_relative_eq!(integral, 6.0, epsilon = 1e-14);

        // 2-point rule integrates x³ exactly (degree 2n-1 = 3)
        // ∫_{-1}^{1} x³ dx = 0 (odd function)
        let rule = gauss_1d(2);
        let integral: f64 = rule.iter().map(|&(x, w)| x.powi(3) * w).sum();
        assert_relative_eq!(integral, 0.0, epsilon = 1e-14);

        // 2-point rule integrates x² exactly
        // ∫_{-1}^{1} x² dx = 2/3
        let integral: f64 = rule.iter().map(|&(x, w)| x.powi(2) * w).sum();
        assert_relative_eq!(integral, 2.0 / 3.0, epsilon = 1e-14);
    }

    #[test]
    fn test_gauss_1d_weights_sum() {
        // Weights should sum to 2 (length of [-1, 1])
        for n in 1..=4 {
            let rule = gauss_1d(n);
            let sum: f64 = rule.iter().map(|&(_, w)| w).sum();
            assert_relative_eq!(sum, 2.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_gauss_tet_weights_sum() {
        // Weights should sum to 1/6 (volume of unit tet)
        for &n in &[1, 4, 5] {
            let rule = gauss_tet(n);
            let sum: f64 = rule.iter().map(|gp| gp.weight).sum();
            assert_relative_eq!(sum, 1.0 / 6.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_gauss_tet_barycentric_sum() {
        // Barycentric coordinates should sum to 1
        for &n in &[1, 4, 5] {
            let rule = gauss_tet(n);
            for gp in &rule {
                let sum: f64 = gp.coords.iter().sum();
                assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_gauss_tet_integrates_constant() {
        // ∫ 1 dV = 1/6 for unit tet
        for &n in &[1, 4, 5] {
            let rule = gauss_tet(n);
            let integral: f64 = rule.iter().map(|gp| gp.weight).sum();
            assert_relative_eq!(integral, 1.0 / 6.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_gauss_hex_point_count() {
        assert_eq!(gauss_hex(1).len(), 1);
        assert_eq!(gauss_hex(2).len(), 8);
        assert_eq!(gauss_hex(3).len(), 27);
    }

    #[test]
    fn test_gauss_hex_weights_sum() {
        // Weights should sum to 8 (volume of [-1,1]³)
        for n in 1..=3 {
            let rule = gauss_hex(n);
            let sum: f64 = rule.iter().map(|gp| gp.weight).sum();
            assert_relative_eq!(sum, 8.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_gauss_tri_weights_sum() {
        // Weights should sum to 1/2 (area of unit triangle)
        for &n in &[1, 3, 4] {
            let rule = gauss_tri(n);
            let sum: f64 = rule.iter().map(|gp| gp.weight).sum();
            assert_relative_eq!(sum, 0.5, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_gauss_quad_weights_sum() {
        // Weights should sum to 4 (area of [-1,1]²)
        for n in 1..=3 {
            let rule = gauss_quad(n);
            let sum: f64 = rule.iter().map(|gp| gp.weight).sum();
            assert_relative_eq!(sum, 4.0, epsilon = 1e-14);
        }
    }

    #[test]
    #[should_panic(expected = "gauss_1d: n must be")]
    fn test_gauss_1d_invalid_n() {
        gauss_1d(5);
    }

    #[test]
    #[should_panic(expected = "gauss_tet: n must be")]
    fn test_gauss_tet_invalid_n() {
        gauss_tet(3);
    }

    #[test]
    #[should_panic(expected = "gauss_hex: n must be")]
    fn test_gauss_hex_invalid_n() {
        gauss_hex(4);
    }
}
