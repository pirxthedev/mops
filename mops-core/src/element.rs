//! Element trait and implementations.
//!
//! The Element trait defines the interface for finite elements, enabling
//! the assembly process to work with any element type uniformly.
//!
//! # Submodules
//!
//! - [`gauss`] - Gauss quadrature rules for numerical integration
//! - [`tet4`] - 4-node tetrahedron (constant strain)
//! - [`tet10`] - 10-node tetrahedron (quadratic)
//! - [`hex8`] - 8-node hexahedron (trilinear brick)
//! - [`hex8_sri`] - 8-node hexahedron with selective reduced integration (shear locking mitigation)
//! - [`hex20`] - 20-node hexahedron (serendipity quadratic brick)
//! - [`plane_stress`] - 2D plane stress elements (Tri3, Quad4)
//! - [`plane_strain`] - 2D plane strain elements (Tri3, Quad4)
//! - [`axisymmetric`] - Axisymmetric elements for bodies of revolution (Tri3, Quad4)
//!
//! # Element Dispatch
//!
//! Use [`create_element`] to instantiate an element implementation from an
//! [`ElementType`](crate::mesh::ElementType):
//!
//! ```
//! use mops_core::element::create_element;
//! use mops_core::mesh::ElementType;
//!
//! let element = create_element(ElementType::Tet4);
//! assert_eq!(element.n_nodes(), 4);
//!
//! let tet10 = create_element(ElementType::Tet10);
//! assert_eq!(tet10.n_nodes(), 10);
//!
//! let hex = create_element(ElementType::Hex8);
//! assert_eq!(hex.n_nodes(), 8);
//! ```

use crate::material::Material;
use crate::mesh::ElementType;
use crate::types::{Point3, StressTensor};
use nalgebra::DMatrix;

pub mod axisymmetric;
pub mod gauss;
pub mod hex20;
pub mod hex8;
pub mod hex8_bbar;
pub mod hex8_sri;
pub mod plane_strain;
pub mod plane_stress;
pub mod quad8;
pub mod tet10;
pub mod tet4;
pub mod tri6;

pub use axisymmetric::{Quad4Axisymmetric, Tri3Axisymmetric};
pub use gauss::{gauss_1d, gauss_hex, gauss_quad, gauss_tet, gauss_tri, GaussPoint};
pub use hex20::Hex20;
pub use hex8::Hex8;
pub use hex8_bbar::Hex8Bbar;
pub use hex8_sri::Hex8SRI;
pub use plane_strain::{Quad4PlaneStrain, Tri3PlaneStrain};
pub use plane_stress::{Quad4, Tri3};
pub use quad8::Quad8;
pub use tet10::Tet10;
pub use tet4::Tet4;
pub use tri6::Tri6;

/// Finite element interface.
///
/// All element types implement this trait, providing:
/// - Element stiffness matrix computation
/// - Stress recovery from nodal displacements
///
/// Elements must be thread-safe (Send + Sync) to enable parallel assembly.
pub trait Element: Send + Sync {
    /// Number of nodes in this element.
    fn n_nodes(&self) -> usize;

    /// Degrees of freedom per node (3 for 3D solid elements).
    fn dofs_per_node(&self) -> usize;

    /// Total degrees of freedom for this element.
    fn n_dofs(&self) -> usize {
        self.n_nodes() * self.dofs_per_node()
    }

    /// Compute the element stiffness matrix.
    ///
    /// # Arguments
    ///
    /// * `coords` - Nodal coordinates, shape (n_nodes,)
    /// * `material` - Material properties
    ///
    /// # Returns
    ///
    /// Dense stiffness matrix of shape (n_dofs, n_dofs)
    fn stiffness(&self, coords: &[Point3], material: &Material) -> DMatrix<f64>;

    /// Compute stress at element integration points.
    ///
    /// # Arguments
    ///
    /// * `coords` - Nodal coordinates
    /// * `displacements` - Nodal displacement vector (length = n_dofs)
    /// * `material` - Material properties
    ///
    /// # Returns
    ///
    /// Stress tensor at each integration point
    fn stress(
        &self,
        coords: &[Point3],
        displacements: &[f64],
        material: &Material,
    ) -> Vec<StressTensor>;

    /// Compute element volume.
    fn volume(&self, coords: &[Point3]) -> f64;
}

/// Create an element implementation from an element type.
///
/// Returns a boxed trait object implementing [`Element`] for the given
/// [`ElementType`]. This enables the assembly loop to work uniformly with
/// any element type.
///
/// For 2D elements that require a thickness parameter, use
/// [`create_element_with_thickness`] instead.
///
/// # Arguments
///
/// * `element_type` - The element type to instantiate
///
/// # Panics
///
/// Panics if the element type is not yet implemented or requires additional
/// parameters (like thickness for 2D elements).
///
/// # Example
///
/// ```
/// use mops_core::element::create_element;
/// use mops_core::mesh::ElementType;
///
/// let element = create_element(ElementType::Tet4);
/// assert_eq!(element.n_nodes(), 4);
/// assert_eq!(element.dofs_per_node(), 3);
/// ```
pub fn create_element(element_type: ElementType) -> Box<dyn Element> {
    match element_type {
        ElementType::Tet4 => Box::new(Tet4::new()),
        ElementType::Tet10 => Box::new(Tet10::new()),
        ElementType::Hex8 => Box::new(Hex8::new()),
        ElementType::Hex8SRI => Box::new(Hex8SRI::new()),
        ElementType::Hex8Bbar => Box::new(Hex8Bbar::new()),
        ElementType::Hex20 => Box::new(Hex20::new()),
        ElementType::Tri3 | ElementType::Tri6 | ElementType::Quad4 | ElementType::Quad8 => {
            // Use default thickness of 1.0 for plane stress elements
            create_element_with_thickness(element_type, 1.0)
        }
    }
}

/// Create a 2D element with specified thickness.
///
/// For 2D plane stress elements (Tri3, Tri6, Quad4, Quad8), the thickness parameter
/// affects the stiffness matrix: K = t * ∫∫ B^T * D * B dA
///
/// # Arguments
///
/// * `element_type` - The 2D element type to instantiate
/// * `thickness` - Element thickness (must be positive)
///
/// # Panics
///
/// Panics if:
/// - The element type is not a 2D plane stress element
/// - The thickness is not positive
///
/// # Example
///
/// ```
/// use mops_core::element::create_element_with_thickness;
/// use mops_core::mesh::ElementType;
///
/// let element = create_element_with_thickness(ElementType::Tri3, 0.1);
/// assert_eq!(element.n_nodes(), 3);
/// assert_eq!(element.dofs_per_node(), 2);
///
/// let tri6 = create_element_with_thickness(ElementType::Tri6, 0.5);
/// assert_eq!(tri6.n_nodes(), 6);
///
/// let quad8 = create_element_with_thickness(ElementType::Quad8, 1.0);
/// assert_eq!(quad8.n_nodes(), 8);
/// ```
pub fn create_element_with_thickness(
    element_type: ElementType,
    thickness: f64,
) -> Box<dyn Element> {
    match element_type {
        ElementType::Tri3 => Box::new(Tri3::new(thickness)),
        ElementType::Tri6 => Box::new(Tri6::new(thickness)),
        ElementType::Quad4 => Box::new(Quad4::new(thickness)),
        ElementType::Quad8 => Box::new(Quad8::new(thickness)),
        _ => panic!(
            "Element type {:?} does not support thickness parameter. \
             Use create_element() for 3D elements.",
            element_type
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_element_tet4() {
        let element = create_element(ElementType::Tet4);
        assert_eq!(element.n_nodes(), 4);
        assert_eq!(element.dofs_per_node(), 3);
        assert_eq!(element.n_dofs(), 12);
    }

    #[test]
    fn test_create_element_tet10() {
        let element = create_element(ElementType::Tet10);
        assert_eq!(element.n_nodes(), 10);
        assert_eq!(element.dofs_per_node(), 3);
        assert_eq!(element.n_dofs(), 30);
    }

    #[test]
    fn test_create_element_hex8() {
        let element = create_element(ElementType::Hex8);
        assert_eq!(element.n_nodes(), 8);
        assert_eq!(element.dofs_per_node(), 3);
        assert_eq!(element.n_dofs(), 24);
    }

    #[test]
    fn test_create_element_hex20() {
        let element = create_element(ElementType::Hex20);
        assert_eq!(element.n_nodes(), 20);
        assert_eq!(element.dofs_per_node(), 3);
        assert_eq!(element.n_dofs(), 60);
    }

    #[test]
    fn test_create_element_hex8_sri() {
        let element = create_element(ElementType::Hex8SRI);
        assert_eq!(element.n_nodes(), 8);
        assert_eq!(element.dofs_per_node(), 3);
        assert_eq!(element.n_dofs(), 24);
    }

    #[test]
    fn test_create_element_hex8_bbar() {
        let element = create_element(ElementType::Hex8Bbar);
        assert_eq!(element.n_nodes(), 8);
        assert_eq!(element.dofs_per_node(), 3);
        assert_eq!(element.n_dofs(), 24);
    }

    #[test]
    fn test_create_element_tri3() {
        let element = create_element(ElementType::Tri3);
        assert_eq!(element.n_nodes(), 3);
        assert_eq!(element.dofs_per_node(), 2);
        assert_eq!(element.n_dofs(), 6);
    }

    #[test]
    fn test_create_element_quad4() {
        let element = create_element(ElementType::Quad4);
        assert_eq!(element.n_nodes(), 4);
        assert_eq!(element.dofs_per_node(), 2);
        assert_eq!(element.n_dofs(), 8);
    }

    #[test]
    fn test_create_element_with_thickness_tri3() {
        let element = create_element_with_thickness(ElementType::Tri3, 0.5);
        assert_eq!(element.n_nodes(), 3);
        assert_eq!(element.dofs_per_node(), 2);
        assert_eq!(element.n_dofs(), 6);
    }

    #[test]
    fn test_create_element_with_thickness_quad4() {
        let element = create_element_with_thickness(ElementType::Quad4, 2.0);
        assert_eq!(element.n_nodes(), 4);
        assert_eq!(element.dofs_per_node(), 2);
        assert_eq!(element.n_dofs(), 8);
    }

    #[test]
    fn test_create_element_tri6() {
        let element = create_element(ElementType::Tri6);
        assert_eq!(element.n_nodes(), 6);
        assert_eq!(element.dofs_per_node(), 2);
        assert_eq!(element.n_dofs(), 12);
    }

    #[test]
    fn test_create_element_quad8() {
        let element = create_element(ElementType::Quad8);
        assert_eq!(element.n_nodes(), 8);
        assert_eq!(element.dofs_per_node(), 2);
        assert_eq!(element.n_dofs(), 16);
    }

    #[test]
    fn test_create_element_with_thickness_tri6() {
        let element = create_element_with_thickness(ElementType::Tri6, 0.5);
        assert_eq!(element.n_nodes(), 6);
        assert_eq!(element.dofs_per_node(), 2);
        assert_eq!(element.n_dofs(), 12);
    }

    #[test]
    fn test_create_element_with_thickness_quad8() {
        let element = create_element_with_thickness(ElementType::Quad8, 2.0);
        assert_eq!(element.n_nodes(), 8);
        assert_eq!(element.dofs_per_node(), 2);
        assert_eq!(element.n_dofs(), 16);
    }
}
