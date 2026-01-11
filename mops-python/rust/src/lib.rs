//! Python bindings for MOPS FEA solver.
//!
//! This crate provides PyO3 bindings exposing mops-core functionality to Python.
//! The bindings follow a minimal wrapper pattern - most logic stays in Rust.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

use mops_core::assembly::{assemble, AssemblyOptions, BoundaryCondition};
use mops_core::element::create_element;
use mops_core::material::Material as CoreMaterial;
use mops_core::mesh::{ElementType, Mesh};
use mops_core::solver::{FaerCholeskySolver, Solver, SolverConfig, SolverType};
use mops_core::stress::{recover_stresses, StressField};
use mops_core::types::Point3;

/// Material definition for Python.
#[pyclass(name = "Material")]
#[derive(Clone)]
pub struct PyMaterial {
    name: String,
    e: f64,
    nu: f64,
    rho: f64,
}

#[pymethods]
impl PyMaterial {
    #[new]
    #[pyo3(signature = (name, e, nu, rho=0.0))]
    fn new(name: String, e: f64, nu: f64, rho: f64) -> PyResult<Self> {
        if e <= 0.0 {
            return Err(PyValueError::new_err("Young's modulus must be positive"));
        }
        if nu <= -1.0 || nu >= 0.5 {
            return Err(PyValueError::new_err(
                "Poisson's ratio must be in (-1, 0.5)",
            ));
        }
        Ok(Self { name, e, nu, rho })
    }

    #[staticmethod]
    fn steel() -> Self {
        Self {
            name: "steel".to_string(),
            e: 200e9,
            nu: 0.3,
            rho: 7850.0,
        }
    }

    #[staticmethod]
    fn aluminum() -> Self {
        Self {
            name: "aluminum".to_string(),
            e: 68.9e9,
            nu: 0.33,
            rho: 2700.0,
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn e(&self) -> f64 {
        self.e
    }

    #[getter]
    fn nu(&self) -> f64 {
        self.nu
    }

    #[getter]
    fn rho(&self) -> f64 {
        self.rho
    }

    fn __repr__(&self) -> String {
        format!(
            "Material({}, E={:.2e}, nu={:.2}, rho={:.1})",
            self.name, self.e, self.nu, self.rho
        )
    }
}

impl PyMaterial {
    fn to_core(&self) -> CoreMaterial {
        CoreMaterial::new(self.e, self.nu).expect("Material already validated")
    }
}

/// Mesh data for Python.
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    inner: Mesh,
    element_type: ElementType,
}

#[pymethods]
impl PyMesh {
    /// Create mesh from numpy arrays.
    ///
    /// Args:
    ///     nodes: Nx3 array of node coordinates
    ///     elements: MxK array of element connectivity (node indices)
    ///     element_type: Element type string ("tet4", "tet10", "hex8")
    #[new]
    fn new(
        nodes: PyReadonlyArray2<f64>,
        elements: PyReadonlyArray2<i64>,
        element_type: &str,
    ) -> PyResult<Self> {
        let elem_type = match element_type {
            "tet4" => ElementType::Tet4,
            "tet10" => ElementType::Tet10,
            "hex8" => ElementType::Hex8,
            "hex20" => ElementType::Hex20,
            "tri3" => ElementType::Tri3,
            "tri6" => ElementType::Tri6,
            "quad4" => ElementType::Quad4,
            "quad8" => ElementType::Quad8,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown element type: {}. Valid types: tet4, tet10, hex8",
                    element_type
                )))
            }
        };

        let nodes_shape = nodes.shape();
        if nodes_shape.len() != 2 || nodes_shape[1] != 3 {
            return Err(PyValueError::new_err("nodes must be Nx3 array"));
        }

        let nodes_array = nodes.as_array();
        let mut mesh = Mesh::with_capacity(nodes_shape[0], 0);

        for i in 0..nodes_shape[0] {
            mesh.add_node(Point3::new(
                nodes_array[[i, 0]],
                nodes_array[[i, 1]],
                nodes_array[[i, 2]],
            ));
        }

        let elements_array = elements.as_array();
        let elements_shape = elements.shape();
        let expected_nodes = elem_type.n_nodes();

        if elements_shape.len() != 2 || elements_shape[1] != expected_nodes {
            return Err(PyValueError::new_err(format!(
                "elements must be Mx{} array for {} elements",
                expected_nodes, element_type
            )));
        }

        for i in 0..elements_shape[0] {
            let node_ids: Vec<usize> = (0..expected_nodes)
                .map(|j| elements_array[[i, j]] as usize)
                .collect();
            mesh.add_element(elem_type, node_ids)
                .map_err(|e| PyValueError::new_err(format!("Element error: {}", e)))?;
        }

        Ok(Self {
            inner: mesh,
            element_type: elem_type,
        })
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    #[getter]
    fn n_elements(&self) -> usize {
        self.inner.n_elements()
    }

    /// Get node coordinates as Nx3 numpy array.
    ///
    /// Returns:
    ///     2D numpy array of shape (n_nodes, 3) containing [x, y, z] coordinates
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let nodes = self.inner.nodes();
        let data: Vec<Vec<f64>> = nodes.iter().map(|p| vec![p[0], p[1], p[2]]).collect();
        PyArray2::from_vec2(py, &data).expect("from_vec2 should succeed")
    }

    /// Get element connectivity as MxK numpy array.
    ///
    /// Returns:
    ///     2D numpy array of shape (n_elements, nodes_per_element) containing node indices
    #[getter]
    fn elements<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let elements = self.inner.elements();
        let data: Vec<Vec<i64>> = elements
            .iter()
            .map(|e| e.nodes.iter().map(|&n| n as i64).collect())
            .collect();
        PyArray2::from_vec2(py, &data).expect("from_vec2 should succeed")
    }

    fn __repr__(&self) -> String {
        format!(
            "Mesh(nodes={}, elements={})",
            self.inner.n_nodes(),
            self.inner.n_elements()
        )
    }
}

/// Solver configuration.
#[pyclass(name = "SolverConfig")]
#[derive(Clone)]
pub struct PySolverConfig {
    solver_type: String,
    auto_threshold: usize,
    tolerance: f64,
    max_iterations: usize,
}

#[pymethods]
impl PySolverConfig {
    #[new]
    #[pyo3(signature = (solver_type="auto", auto_threshold=100_000, tolerance=1e-10, max_iterations=1000))]
    fn new(
        solver_type: &str,
        auto_threshold: usize,
        tolerance: f64,
        max_iterations: usize,
    ) -> PyResult<Self> {
        match solver_type {
            "auto" | "direct" | "iterative" => {}
            _ => {
                return Err(PyValueError::new_err(
                    "solver_type must be 'auto', 'direct', or 'iterative'",
                ))
            }
        }
        Ok(Self {
            solver_type: solver_type.to_string(),
            auto_threshold,
            tolerance,
            max_iterations,
        })
    }
}

impl PySolverConfig {
    fn to_core(&self) -> SolverConfig {
        SolverConfig {
            solver_type: match self.solver_type.as_str() {
                "direct" => SolverType::Direct,
                "iterative" => SolverType::Iterative,
                _ => SolverType::Auto,
            },
            auto_threshold: self.auto_threshold,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
        }
    }
}

/// FEA solution results.
#[pyclass(name = "Results")]
pub struct PyResults {
    displacements: Vec<f64>,
    n_nodes: usize,
    n_elements: usize,
    /// Element stress data: for each element, the average stress tensor (6 components).
    element_stresses: Vec<[f64; 6]>,
    /// Von Mises stress per element.
    von_mises_stresses: Vec<f64>,
}

#[pymethods]
impl PyResults {
    /// Get displacement array (n_nodes x 3).
    fn displacement<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        // Reshape displacements into Nx3 nested vec
        let data: Vec<Vec<f64>> = (0..self.n_nodes)
            .map(|i| {
                vec![
                    self.displacements.get(i * 3).copied().unwrap_or(0.0),
                    self.displacements.get(i * 3 + 1).copied().unwrap_or(0.0),
                    self.displacements.get(i * 3 + 2).copied().unwrap_or(0.0),
                ]
            })
            .collect();
        PyArray2::from_vec2(py, &data).expect("from_vec2 should succeed")
    }

    /// Get displacement magnitude array.
    fn displacement_magnitude<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let magnitudes: Vec<f64> = (0..self.n_nodes)
            .map(|i| {
                let ux = self.displacements.get(i * 3).copied().unwrap_or(0.0);
                let uy = self.displacements.get(i * 3 + 1).copied().unwrap_or(0.0);
                let uz = self.displacements.get(i * 3 + 2).copied().unwrap_or(0.0);
                (ux * ux + uy * uy + uz * uz).sqrt()
            })
            .collect();
        PyArray1::from_vec(py, magnitudes)
    }

    /// Get maximum displacement magnitude.
    fn max_displacement(&self) -> f64 {
        (0..self.n_nodes)
            .map(|i| {
                let ux = self.displacements.get(i * 3).copied().unwrap_or(0.0);
                let uy = self.displacements.get(i * 3 + 1).copied().unwrap_or(0.0);
                let uz = self.displacements.get(i * 3 + 2).copied().unwrap_or(0.0);
                (ux * ux + uy * uy + uz * uz).sqrt()
            })
            .fold(0.0, f64::max)
    }

    /// Get element stress tensors (n_elements x 6).
    ///
    /// Each row contains the average stress tensor components for one element
    /// in Voigt notation: [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]
    fn stress<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let data: Vec<Vec<f64>> = self.element_stresses.iter().map(|s| s.to_vec()).collect();
        PyArray2::from_vec2(py, &data).expect("from_vec2 should succeed")
    }

    /// Get von Mises stress per element (n_elements,).
    fn von_mises<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.von_mises_stresses.clone())
    }

    /// Get maximum von Mises stress across all elements.
    fn max_von_mises(&self) -> f64 {
        self.von_mises_stresses.iter().copied().fold(0.0, f64::max)
    }

    /// Get stress for a specific element by index.
    ///
    /// Returns the average stress tensor [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]
    fn element_stress<'py>(
        &self,
        py: Python<'py>,
        element_id: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if element_id >= self.n_elements {
            return Err(PyValueError::new_err(format!(
                "Element index {} out of bounds (n_elements={})",
                element_id, self.n_elements
            )));
        }
        Ok(PyArray1::from_vec(
            py,
            self.element_stresses[element_id].to_vec(),
        ))
    }

    /// Get element von Mises stress by element index.
    fn element_von_mises(&self, element_id: usize) -> PyResult<f64> {
        if element_id >= self.n_elements {
            return Err(PyValueError::new_err(format!(
                "Element index {} out of bounds (n_elements={})",
                element_id, self.n_elements
            )));
        }
        Ok(self.von_mises_stresses[element_id])
    }

    fn __repr__(&self) -> String {
        format!(
            "Results(n_nodes={}, n_elements={}, max_disp={:.3e}, max_vm={:.3e})",
            self.n_nodes,
            self.n_elements,
            self.max_displacement(),
            self.max_von_mises()
        )
    }
}

/// Solve a simple FEA problem.
///
/// This is a convenience function for testing the solver pipeline.
#[pyfunction]
#[pyo3(signature = (mesh, material, constrained_nodes, loaded_nodes, load_vector, _config=None))]
fn solve_simple(
    mesh: &PyMesh,
    material: &PyMaterial,
    constrained_nodes: PyReadonlyArray1<i64>,
    loaded_nodes: PyReadonlyArray1<i64>,
    load_vector: PyReadonlyArray1<f64>,
    _config: Option<&PySolverConfig>,
) -> PyResult<PyResults> {
    let core_material = material.to_core();

    // Build boundary conditions
    let constrained = constrained_nodes.as_array();
    let loaded = loaded_nodes.as_array();
    let load_vec = load_vector.as_array();

    if load_vec.len() != 3 {
        return Err(PyValueError::new_err("load_vector must have 3 components"));
    }

    let mut bcs = Vec::new();

    // Add displacement constraints (fix all DOFs at constrained nodes)
    for &node_idx in constrained.iter() {
        let node = node_idx as usize;
        for dof in 0..3 {
            bcs.push(BoundaryCondition::Displacement {
                node,
                dof,
                value: 0.0,
            });
        }
    }

    // Add forces at loaded nodes
    for &node_idx in loaded.iter() {
        let node = node_idx as usize;
        for dof in 0..3 {
            if load_vec[dof].abs() > 1e-15 {
                bcs.push(BoundaryCondition::Force {
                    node,
                    dof,
                    value: load_vec[dof],
                });
            }
        }
    }

    // Assemble the system
    let options = AssemblyOptions::default();
    let system = assemble(&mesh.inner, &core_material, &bcs, &options)
        .map_err(|e| PyRuntimeError::new_err(format!("Assembly error: {}", e)))?;

    // Apply constraints by elimination
    // Build reduced system excluding constrained DOFs
    let n_dofs = system.n_dofs;
    let free_dofs: Vec<usize> = (0..n_dofs)
        .filter(|dof| !system.constraints.contains_key(dof))
        .collect();

    if free_dofs.is_empty() {
        // All DOFs constrained - solution is prescribed values
        let mut displacements = vec![0.0; n_dofs];
        for (&dof, &value) in &system.constraints {
            displacements[dof] = value;
        }
        // Recover stresses from constrained displacements
        let stress_field = recover_stresses(&mesh.inner, &core_material, &displacements);
        let (element_stresses, von_mises_stresses) = extract_stress_data(&stress_field);
        return Ok(PyResults {
            displacements,
            n_nodes: mesh.inner.n_nodes(),
            n_elements: mesh.inner.n_elements(),
            element_stresses,
            von_mises_stresses,
        });
    }

    // Create mapping from free DOF indices to reduced indices
    let mut dof_to_reduced: HashMap<usize, usize> = HashMap::new();
    for (reduced_idx, &dof) in free_dofs.iter().enumerate() {
        dof_to_reduced.insert(dof, reduced_idx);
    }

    // Build reduced system
    let n_free = free_dofs.len();
    let mut reduced_triplet = mops_core::sparse::TripletMatrix::new(n_free, n_free);
    let mut reduced_rhs = vec![0.0; n_free];

    // Copy relevant entries from full stiffness matrix
    let k = &system.stiffness;
    let row_offsets = k.row_offsets();
    let col_indices = k.col_indices();
    let values = k.values();

    for row in 0..k.nrows() {
        if let Some(&reduced_row) = dof_to_reduced.get(&row) {
            for idx in row_offsets[row]..row_offsets[row + 1] {
                let col = col_indices[idx];
                if let Some(&reduced_col) = dof_to_reduced.get(&col) {
                    reduced_triplet.add(reduced_row, reduced_col, values[idx]);
                }
            }
            reduced_rhs[reduced_row] = system.rhs[row];
        }
    }

    let reduced_matrix = reduced_triplet.to_csr();

    // Solve the reduced system
    let solver = FaerCholeskySolver::new();
    let reduced_solution = solver
        .solve(&reduced_matrix, &reduced_rhs)
        .map_err(|e| PyRuntimeError::new_err(format!("Solver error: {}", e)))?;

    // Reconstruct full displacement vector
    let mut displacements = vec![0.0; n_dofs];
    for (reduced_idx, &dof) in free_dofs.iter().enumerate() {
        displacements[dof] = reduced_solution[reduced_idx];
    }
    for (&dof, &value) in &system.constraints {
        displacements[dof] = value;
    }

    // Recover stresses from displacement solution
    let stress_field = recover_stresses(&mesh.inner, &core_material, &displacements);
    let (element_stresses, von_mises_stresses) = extract_stress_data(&stress_field);

    Ok(PyResults {
        displacements,
        n_nodes: mesh.inner.n_nodes(),
        n_elements: mesh.inner.n_elements(),
        element_stresses,
        von_mises_stresses,
    })
}

/// Extract stress data from StressField into arrays for PyResults.
fn extract_stress_data(stress_field: &StressField) -> (Vec<[f64; 6]>, Vec<f64>) {
    let element_stresses: Vec<[f64; 6]> = stress_field
        .element_stresses
        .iter()
        .map(|es| {
            let avg = es.average_stress();
            [avg.0[0], avg.0[1], avg.0[2], avg.0[3], avg.0[4], avg.0[5]]
        })
        .collect();

    let von_mises_stresses: Vec<f64> = stress_field
        .element_stresses
        .iter()
        .map(|es| es.average_stress().von_mises())
        .collect();

    (element_stresses, von_mises_stresses)
}

/// Compute element stiffness matrix for a single element.
///
/// This function is primarily for testing and verification purposes.
/// It computes the element stiffness matrix K for a single element
/// given its nodal coordinates and material properties.
///
/// Args:
///     element_type: Element type string ("tet4", "tet10", "hex8")
///     nodes: Nx3 array of node coordinates for this element
///     material: Material properties
///
/// Returns:
///     2D numpy array of shape (n_dofs, n_dofs) containing the element stiffness matrix
#[pyfunction]
fn element_stiffness<'py>(
    py: Python<'py>,
    element_type: &str,
    nodes: PyReadonlyArray2<f64>,
    material: &PyMaterial,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let elem_type = match element_type {
        "tet4" => ElementType::Tet4,
        "tet10" => ElementType::Tet10,
        "hex8" => ElementType::Hex8,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown element type: {}. Valid types: tet4, tet10, hex8",
                element_type
            )))
        }
    };

    let nodes_shape = nodes.shape();
    let expected_nodes = elem_type.n_nodes();

    if nodes_shape.len() != 2 || nodes_shape[0] != expected_nodes || nodes_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "nodes must be {}x3 array for {} element",
            expected_nodes, element_type
        )));
    }

    // Convert nodes to Point3 array
    let nodes_array = nodes.as_array();
    let coords: Vec<Point3> = (0..expected_nodes)
        .map(|i| {
            Point3::new(
                nodes_array[[i, 0]],
                nodes_array[[i, 1]],
                nodes_array[[i, 2]],
            )
        })
        .collect();

    // Create element and compute stiffness
    let element = create_element(elem_type);
    let core_material = material.to_core();
    let k = element.stiffness(&coords, &core_material);

    // Convert to 2D nested vec for PyArray2
    let n_dofs = element.n_dofs();
    let data: Vec<Vec<f64>> = (0..n_dofs)
        .map(|i| (0..n_dofs).map(|j| k[(i, j)]).collect())
        .collect();

    PyArray2::from_vec2(py, &data)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create array: {}", e)))
}

/// Compute element volume for a single element.
///
/// Args:
///     element_type: Element type string ("tet4", "tet10", "hex8")
///     nodes: Nx3 array of node coordinates for this element
///
/// Returns:
///     Element volume as a float
#[pyfunction]
fn element_volume(element_type: &str, nodes: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let elem_type = match element_type {
        "tet4" => ElementType::Tet4,
        "tet10" => ElementType::Tet10,
        "hex8" => ElementType::Hex8,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown element type: {}. Valid types: tet4, tet10, hex8",
                element_type
            )))
        }
    };

    let nodes_shape = nodes.shape();
    let expected_nodes = elem_type.n_nodes();

    if nodes_shape.len() != 2 || nodes_shape[0] != expected_nodes || nodes_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "nodes must be {}x3 array for {} element",
            expected_nodes, element_type
        )));
    }

    // Convert nodes to Point3 array
    let nodes_array = nodes.as_array();
    let coords: Vec<Point3> = (0..expected_nodes)
        .map(|i| {
            Point3::new(
                nodes_array[[i, 0]],
                nodes_array[[i, 1]],
                nodes_array[[i, 2]],
            )
        })
        .collect();

    // Create element and compute volume
    let element = create_element(elem_type);
    Ok(element.volume(&coords))
}

/// Compute stress tensor for a single element given displacements.
///
/// This function computes the stress at the integration points of a single element
/// given its nodal coordinates, nodal displacements, and material properties.
///
/// Args:
///     element_type: Element type string ("tet4", "tet10", "hex8")
///     nodes: Nx3 array of node coordinates for this element
///     displacements: Flat array of nodal displacements (N*3 elements: [u0, v0, w0, u1, v1, w1, ...])
///     material: Material properties
///
/// Returns:
///     2D numpy array of shape (n_integration_points, 6) containing stress tensors
///     in Voigt notation: [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]
#[pyfunction]
fn compute_element_stress<'py>(
    py: Python<'py>,
    element_type: &str,
    nodes: PyReadonlyArray2<f64>,
    displacements: PyReadonlyArray1<f64>,
    material: &PyMaterial,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let elem_type = match element_type {
        "tet4" => ElementType::Tet4,
        "tet10" => ElementType::Tet10,
        "hex8" => ElementType::Hex8,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown element type: {}. Valid types: tet4, tet10, hex8",
                element_type
            )))
        }
    };

    let nodes_shape = nodes.shape();
    let expected_nodes = elem_type.n_nodes();
    let expected_dofs = expected_nodes * 3;

    if nodes_shape.len() != 2 || nodes_shape[0] != expected_nodes || nodes_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "nodes must be {}x3 array for {} element",
            expected_nodes, element_type
        )));
    }

    let disp_array = displacements.as_array();
    if disp_array.len() != expected_dofs {
        return Err(PyValueError::new_err(format!(
            "displacements must have {} elements for {} element",
            expected_dofs, element_type
        )));
    }

    // Convert nodes to Point3 array
    let nodes_array = nodes.as_array();
    let coords: Vec<Point3> = (0..expected_nodes)
        .map(|i| {
            Point3::new(
                nodes_array[[i, 0]],
                nodes_array[[i, 1]],
                nodes_array[[i, 2]],
            )
        })
        .collect();

    // Convert displacements to slice
    let disp_vec: Vec<f64> = disp_array.iter().copied().collect();

    // Create element and compute stress
    let element = create_element(elem_type);
    let core_material = material.to_core();
    let stresses = element.stress(&coords, &disp_vec, &core_material);

    // Convert to 2D nested vec for PyArray2
    let data: Vec<Vec<f64>> = stresses
        .iter()
        .map(|s| vec![s.0[0], s.0[1], s.0[2], s.0[3], s.0[4], s.0[5]])
        .collect();

    PyArray2::from_vec2(py, &data)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create array: {}", e)))
}

/// Check solver library availability.
#[pyfunction]
fn solver_info() -> HashMap<String, bool> {
    let mut info = HashMap::new();
    info.insert("faer_cholesky".to_string(), true);
    info.insert("dense_lu".to_string(), true);
    // hypre and suitesparse will be added later
    info.insert("hypre".to_string(), false);
    info.insert("suitesparse".to_string(), false);
    info
}

/// Get library version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module definition.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMaterial>()?;
    m.add_class::<PyMesh>()?;
    m.add_class::<PySolverConfig>()?;
    m.add_class::<PyResults>()?;
    m.add_function(wrap_pyfunction!(solve_simple, m)?)?;
    m.add_function(wrap_pyfunction!(element_stiffness, m)?)?;
    m.add_function(wrap_pyfunction!(element_volume, m)?)?;
    m.add_function(wrap_pyfunction!(compute_element_stress, m)?)?;
    m.add_function(wrap_pyfunction!(solver_info, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
