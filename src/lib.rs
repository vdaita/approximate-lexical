mod global_index;
mod kmeans;
mod sparse_to_dense;
mod term_index;
mod utils;

pub use crate::global_index::GlobalIndex;
pub use crate::utils::ApproximateLexicalParameters;
use pyo3::prelude::*;
use serde_json;

#[pyfunction]
pub fn build_approx_index(
    data: Vec<Vec<(usize, f32)>>,
    parameters: ApproximateLexicalParameters,
) -> PyResult<GlobalIndex> {
    let index = GlobalIndex::new(data, parameters);
    Ok(index)
}

#[pyfunction]
pub fn query_approx_index(
    global_index: &GlobalIndex,
    query: Vec<(usize, f32)>,
    num_clusters: usize,
    top_k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    let results = global_index.query(&query, num_clusters, top_k);
    Ok(results)
}

#[pyfunction]
pub fn save_approx_index_to_file(global_index: &GlobalIndex, file_path: String) -> PyResult<()> {
    let serialized = serde_json::to_string(global_index).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Serialization error: {}", e))
    })?;
    std::fs::write(file_path, serialized);
    Ok(())
}

#[pyfunction]
pub fn load_approx_index_from_file(file_path: String) -> PyResult<GlobalIndex> {
    let serialized = std::fs::read_to_string(file_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("File read error: {}", e))
    })?;
    let index: GlobalIndex = serde_json::from_str(&serialized).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Deserialization error: {}", e))
    })?;
    Ok(index)
}

#[pymodule]
pub fn approximate_lexical(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ApproximateLexicalParameters>()?;
    m.add_function(wrap_pyfunction!(build_approx_index, m)?)?;
    m.add_function(wrap_pyfunction!(query_approx_index, m)?)?;
    m.add_function(wrap_pyfunction!(save_approx_index_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(load_approx_index_from_file, m)?);

    // Register other classes and functions as needed
    Ok(())
}
