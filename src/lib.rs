mod global_index;
mod sparse_to_dense;
mod term_index;
mod utils;
mod kmeans;

use pyo3::prelude::*;
use crate::utils::ApproximateLexicalParameters;
use crate::global_index::GlobalIndex;

#[pyfunction]
fn build_index(
    data: Vec<Vec<(usize, f32)>>,
    parameters: ApproximateLexicalParameters
) -> PyResult<GlobalIndex> {
    let index = GlobalIndex::new(data, parameters);
    Ok(index)
}

#[pyfunction]
fn query_index(
    global_index: &GlobalIndex,
    query: Vec<(usize, f32)>,
    num_clusters: usize,
    top_k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    let results = global_index.query(&query, num_clusters, top_k);
    Ok(results)
}

#[pymodule]
fn approximate_lexical(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ApproximateLexicalParameters>()?;
    m.add_function(wrap_pyfunction!(build_index, m)?)?;
    m.add_function(wrap_pyfunction!(query_index, m)?)?;
    
    // Register other classes and functions as needed
    Ok(())
}