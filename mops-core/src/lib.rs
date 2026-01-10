//! MOPS Core - Modular Open Physics Solver
//!
//! High-performance finite element analysis library with:
//! - Element library for 2D and 3D structural elements
//! - Parallel assembly using Rayon
//! - Sparse matrix operations (CSR format)
//! - Direct and iterative linear solvers
//!
//! # Architecture
//!
//! The solver is designed around these core abstractions:
//!
//! - [`Element`] trait: Defines element stiffness and stress recovery
//! - [`Mesh`]: Connectivity and nodal coordinates
//! - [`Material`]: Material property definitions
//! - [`Solver`] trait: Linear system solution strategies

pub mod types;
pub mod element;
pub mod mesh;
pub mod material;
pub mod sparse;
pub mod assembly;
pub mod solver;
pub mod error;

pub use types::{Point3, StressTensor, StrainTensor};
pub use element::{Element, create_element};
pub use mesh::Mesh;
pub use material::Material;
pub use sparse::CsrMatrix;
pub use solver::Solver;
pub use error::{Error, Result};
