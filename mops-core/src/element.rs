//! Element trait and implementations.
//!
//! The Element trait defines the interface for finite elements, enabling
//! the assembly process to work with any element type uniformly.
//!
//! # Submodules
//!
//! - [`gauss`] - Gauss quadrature rules for numerical integration
//! - [`tet4`] - 4-node tetrahedron (constant strain)
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
//! ```

use crate::material::Material;
use crate::mesh::ElementType;
use crate::types::{Point3, StressTensor};
use nalgebra::DMatrix;

pub mod gauss;
pub mod tet4;

// Element implementations (to be added)
// pub mod tet10;
// pub mod hex8;
// pub mod hex20;
// pub mod plane_stress;
// pub mod plane_strain;
// pub mod axisymmetric;

pub use gauss::{gauss_1d, gauss_hex, gauss_quad, gauss_tet, gauss_tri, GaussPoint};
pub use tet4::Tet4;

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
/// # Arguments
///
/// * `element_type` - The element type to instantiate
///
/// # Panics
///
/// Panics if the element type is not yet implemented.
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
        // Future implementations:
        // ElementType::Tet10 => Box::new(Tet10::new()),
        // ElementType::Hex8 => Box::new(Hex8::new()),
        // ElementType::Hex20 => Box::new(Hex20::new()),
        _ => unimplemented!(
            "Element type {:?} is not yet implemented. \
             Currently supported: Tet4",
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
    #[should_panic(expected = "not yet implemented")]
    fn test_create_element_unimplemented_tet10() {
        let _ = create_element(ElementType::Tet10);
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_create_element_unimplemented_hex8() {
        let _ = create_element(ElementType::Hex8);
    }
}
