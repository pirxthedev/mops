//! Mesh data structure for FEA.
//!
//! Stores nodal coordinates and element connectivity.

use crate::error::{Error, Result};
use crate::types::Point3;

/// Element connectivity - node indices for an element.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementConnectivity {
    /// Element type identifier.
    pub element_type: ElementType,
    /// Node indices (0-based).
    pub nodes: Vec<usize>,
}

/// Supported element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// 4-node tetrahedron (linear).
    Tet4,
    /// 10-node tetrahedron (quadratic).
    Tet10,
    /// 8-node hexahedron (linear).
    Hex8,
    /// 20-node hexahedron (quadratic).
    Hex20,
    /// 3-node triangle (plane stress/strain).
    Tri3,
    /// 6-node triangle (quadratic).
    Tri6,
    /// 4-node quadrilateral (plane stress/strain).
    Quad4,
    /// 8-node quadrilateral (quadratic).
    Quad8,
}

impl ElementType {
    /// Number of nodes for this element type.
    pub fn n_nodes(self) -> usize {
        match self {
            ElementType::Tet4 => 4,
            ElementType::Tet10 => 10,
            ElementType::Hex8 => 8,
            ElementType::Hex20 => 20,
            ElementType::Tri3 => 3,
            ElementType::Tri6 => 6,
            ElementType::Quad4 => 4,
            ElementType::Quad8 => 8,
        }
    }

    /// Spatial dimension (2D or 3D).
    pub fn dimension(self) -> usize {
        match self {
            ElementType::Tet4 | ElementType::Tet10 | ElementType::Hex8 | ElementType::Hex20 => 3,
            ElementType::Tri3 | ElementType::Tri6 | ElementType::Quad4 | ElementType::Quad8 => 2,
        }
    }

    /// Degrees of freedom per node for this element type.
    ///
    /// - 3D solid elements: 3 DOFs (u, v, w displacements)
    /// - 2D plane elements: 2 DOFs (u, v displacements)
    pub fn dofs_per_node(self) -> usize {
        match self.dimension() {
            2 => 2,
            3 => 3,
            _ => unreachable!(),
        }
    }
}

/// Finite element mesh.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Nodal coordinates.
    nodes: Vec<Point3>,
    /// Element connectivity.
    elements: Vec<ElementConnectivity>,
}

impl Mesh {
    /// Create a new empty mesh.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            elements: Vec::new(),
        }
    }

    /// Create a mesh with pre-allocated capacity.
    pub fn with_capacity(n_nodes: usize, n_elements: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(n_nodes),
            elements: Vec::with_capacity(n_elements),
        }
    }

    /// Add a node to the mesh, returning its index.
    pub fn add_node(&mut self, point: Point3) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(point);
        idx
    }

    /// Add multiple nodes at once.
    pub fn add_nodes(&mut self, points: impl IntoIterator<Item = Point3>) {
        self.nodes.extend(points);
    }

    /// Add an element to the mesh.
    pub fn add_element(&mut self, element_type: ElementType, nodes: Vec<usize>) -> Result<usize> {
        // Validate node count
        if nodes.len() != element_type.n_nodes() {
            return Err(Error::Mesh(format!(
                "Element type {:?} requires {} nodes, got {}",
                element_type,
                element_type.n_nodes(),
                nodes.len()
            )));
        }

        // Validate node indices
        for &node_idx in &nodes {
            if node_idx >= self.nodes.len() {
                return Err(Error::Mesh(format!(
                    "Node index {} out of bounds (mesh has {} nodes)",
                    node_idx,
                    self.nodes.len()
                )));
            }
        }

        let idx = self.elements.len();
        self.elements.push(ElementConnectivity {
            element_type,
            nodes,
        });
        Ok(idx)
    }

    /// Number of nodes in the mesh.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of elements in the mesh.
    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    /// Get nodal coordinates.
    pub fn nodes(&self) -> &[Point3] {
        &self.nodes
    }

    /// Get a specific node's coordinates.
    pub fn node(&self, idx: usize) -> Option<&Point3> {
        self.nodes.get(idx)
    }

    /// Get element connectivity.
    pub fn elements(&self) -> &[ElementConnectivity] {
        &self.elements
    }

    /// Get a specific element's connectivity.
    pub fn element(&self, idx: usize) -> Option<&ElementConnectivity> {
        self.elements.get(idx)
    }

    /// Get coordinates for an element's nodes.
    pub fn element_coords(&self, elem_idx: usize) -> Option<Vec<Point3>> {
        let elem = self.elements.get(elem_idx)?;
        Some(elem.nodes.iter().map(|&i| self.nodes[i]).collect())
    }

    /// Compute mesh bounding box.
    pub fn bounds(&self) -> Option<(Point3, Point3)> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut min = self.nodes[0];
        let mut max = self.nodes[0];

        for node in &self.nodes[1..] {
            for i in 0..3 {
                min[i] = min[i].min(node[i]);
                max[i] = max[i].max(node[i]);
            }
        }

        Some((min, max))
    }

    /// Determine degrees of freedom per node from mesh element types.
    ///
    /// Returns `Ok(dofs_per_node)` if all elements have the same DOF count,
    /// or `Err` if:
    /// - The mesh has no elements (defaults to 3 for backwards compatibility)
    /// - Elements have inconsistent DOF counts (mixed 2D/3D)
    ///
    /// # Example
    ///
    /// ```
    /// use mops_core::mesh::{Mesh, ElementType};
    /// use nalgebra::Vector3;
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
    /// mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
    /// mesh.add_node(Vector3::new(0.0, 1.0, 0.0));
    /// mesh.add_node(Vector3::new(0.0, 0.0, 1.0));
    /// mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3]).unwrap();
    ///
    /// assert_eq!(mesh.dofs_per_node().unwrap(), 3);
    /// ```
    pub fn dofs_per_node(&self) -> Result<usize> {
        if self.elements.is_empty() {
            // Default to 3D for empty mesh (backwards compatibility)
            return Ok(3);
        }

        let first_dofs = self.elements[0].element_type.dofs_per_node();

        // Check all elements have consistent DOFs
        for (idx, elem) in self.elements.iter().enumerate().skip(1) {
            let elem_dofs = elem.element_type.dofs_per_node();
            if elem_dofs != first_dofs {
                return Err(Error::Mesh(format!(
                    "Inconsistent DOFs per node: element 0 has {} DOFs, element {} has {} DOFs. \
                     Mixed 2D/3D meshes are not supported.",
                    first_dofs, idx, elem_dofs
                )));
            }
        }

        Ok(first_dofs)
    }

    /// Total number of degrees of freedom in the mesh.
    ///
    /// This is `n_nodes * dofs_per_node`.
    ///
    /// # Errors
    ///
    /// Returns an error if elements have inconsistent DOF counts.
    pub fn n_dofs(&self) -> Result<usize> {
        Ok(self.n_nodes() * self.dofs_per_node()?)
    }

    /// Spatial dimension of the mesh (2 or 3).
    ///
    /// Determined from element types. Returns 3 for empty meshes.
    pub fn dimension(&self) -> usize {
        if self.elements.is_empty() {
            return 3;
        }
        self.elements[0].element_type.dimension()
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_mesh_creation() {
        let mut mesh = Mesh::new();

        // Add nodes for a tetrahedron
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 0.0, 1.0));

        assert_eq!(mesh.n_nodes(), 4);

        // Add tet4 element
        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3])
            .unwrap();
        assert_eq!(mesh.n_elements(), 1);
    }

    #[test]
    fn test_invalid_element_node_count() {
        let mut mesh = Mesh::new();
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));

        // Tet4 needs 4 nodes, we only provide 3
        let result = mesh.add_element(ElementType::Tet4, vec![0, 1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_node_index() {
        let mut mesh = Mesh::new();
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));

        // Node index 3 doesn't exist
        let result = mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bounds() {
        let mut mesh = Mesh::new();
        mesh.add_node(Vector3::new(-1.0, -2.0, -3.0));
        mesh.add_node(Vector3::new(1.0, 2.0, 3.0));
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));

        let (min, max) = mesh.bounds().unwrap();
        assert_eq!(min, Vector3::new(-1.0, -2.0, -3.0));
        assert_eq!(max, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_dofs_per_node_3d() {
        let mut mesh = Mesh::new();
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3]).unwrap();

        assert_eq!(mesh.dofs_per_node().unwrap(), 3);
        assert_eq!(mesh.n_dofs().unwrap(), 12);
        assert_eq!(mesh.dimension(), 3);
    }

    #[test]
    fn test_dofs_per_node_2d() {
        let mut mesh = Mesh::new();
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));
        mesh.add_element(ElementType::Tri3, vec![0, 1, 2]).unwrap();

        assert_eq!(mesh.dofs_per_node().unwrap(), 2);
        assert_eq!(mesh.n_dofs().unwrap(), 6);
        assert_eq!(mesh.dimension(), 2);
    }

    #[test]
    fn test_dofs_per_node_empty_mesh() {
        let mesh = Mesh::new();
        // Empty mesh defaults to 3D
        assert_eq!(mesh.dofs_per_node().unwrap(), 3);
        assert_eq!(mesh.dimension(), 3);
    }

    #[test]
    fn test_dofs_per_node_mixed_error() {
        let mut mesh = Mesh::new();
        mesh.add_node(Vector3::new(0.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(1.0, 0.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 1.0, 0.0));
        mesh.add_node(Vector3::new(0.0, 0.0, 1.0));

        // Add a 2D element first
        mesh.add_element(ElementType::Tri3, vec![0, 1, 2]).unwrap();
        // Add a 3D element (uses same nodes)
        mesh.add_element(ElementType::Tet4, vec![0, 1, 2, 3]).unwrap();

        // Should fail due to mixed dimensions
        assert!(mesh.dofs_per_node().is_err());
    }

    #[test]
    fn test_element_type_dofs_per_node() {
        assert_eq!(ElementType::Tet4.dofs_per_node(), 3);
        assert_eq!(ElementType::Tet10.dofs_per_node(), 3);
        assert_eq!(ElementType::Hex8.dofs_per_node(), 3);
        assert_eq!(ElementType::Hex20.dofs_per_node(), 3);
        assert_eq!(ElementType::Tri3.dofs_per_node(), 2);
        assert_eq!(ElementType::Tri6.dofs_per_node(), 2);
        assert_eq!(ElementType::Quad4.dofs_per_node(), 2);
        assert_eq!(ElementType::Quad8.dofs_per_node(), 2);
    }
}
