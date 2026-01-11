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

pub mod assembly;
pub mod element;
pub mod error;
pub mod material;
pub mod mesh;
pub mod solver;
pub mod sparse;
pub mod stress;
pub mod types;

pub use element::{create_element, Element, Quad4, Quad4PlaneStrain, Tri3, Tri3PlaneStrain};
pub use error::{Error, Result};
pub use material::Material;
pub use mesh::Mesh;
pub use solver::{
    select_solver, CachedCholeskySolver, FaerCholeskySolver, Solver, SolverConfig, SolverType,
};
pub use sparse::CsrMatrix;
pub use stress::{compute_nodal_stresses, recover_stresses, ElementStress, StressField};
pub use types::{Point3, StrainTensor, StressTensor};
