mod weighted_lsh_searcher;
mod random_proj;
mod utils;

use std::collections::{HashMap};
use pyo3::prelude::*;
use rand::rng;
use rand_distr::{Distribution, StandardNormal};
use crate::weighted_lsh_searcher::WeightedMinHashLSH;
use crate::random_proj::{RandomProjTermCluster};

// TODO: implement SimHash if Weighted MinHash isn't sufficient
fn sparse_inner_product(query_vec: &[(usize, f32)], doc_vec: &[(usize, f32)]) -> f32 {
    let mut score = 0.0;
    let mut dv_idx = 0;
    let mut q_idx = 0;
    let mut query_sorted = query_vec.to_vec();
    let mut doc_sorted = doc_vec.to_vec();

    query_sorted.sort_by_key(|x| x.0);
    doc_sorted.sort_by_key(|x| x.0);

    while dv_idx < doc_sorted.len() && q_idx < query_sorted.len() {
        if doc_sorted[dv_idx].0 == query_sorted[q_idx].0 {
            score += doc_sorted[dv_idx].1 * query_sorted[q_idx].1;
            dv_idx += 1;
            q_idx += 1;
        } else if doc_sorted[dv_idx].0 < query_sorted[q_idx].0 {
            dv_idx += 1;
        } else {
            q_idx += 1;
        }
    }
    score
}

// --- Index ---
#[pyclass]
#[derive(Clone)]
struct Index {
    lsh: WeightedMinHashLSH,
    
    doc_list: Vec<Vec<(usize, f32)>>,
    vocab_size: usize,
    idf_weights: Vec<f32>
}

impl Index {
    fn build(tokenized_documents: Vec<Vec<usize>>, vocab_size: usize, idf_weights: Vec<f32>, micro: bool) -> Self {
        if micro {
            println!("[micro] Index::build called");
        }
        let scored_documents: Vec<Vec<(usize, f32)>> = tokenized_documents
            .into_iter()
            .map(|tokens| convert_token_list_to_scored_list(tokens, &idf_weights))
            .collect();
        
        let lsh = WeightedMinHashLSH::new(
            &scored_documents,
            8, // num_bands
            42, // seed
            micro
        );

        Self {
            lsh,
            doc_list: scored_documents,
            idf_weights,
            vocab_size,
        }
    }

    fn search(
        &self,
        tokenized_query: Vec<usize>,
        top_k: usize,
        micro: bool,
    ) -> Vec<usize> {
        if micro {
            println!("[micro] Index::search called");
        }
        let query_vec = convert_token_list_to_scored_list(tokenized_query, &self.idf_weights);
        let candidates = self.lsh.query(query_vec.clone());
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&doc_idx| {
                let doc_vec = &self.doc_list[doc_idx];
                let score = sparse_inner_product(&query_vec, doc_vec);
                if micro {
                    println!("[micro] Doc {} score: {}", doc_idx, score);
                }
                (doc_idx, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_k);
        if micro {
            println!("[micro] Top {} results: {:?}", top_k, scored.iter().map(|(idx, _)| idx).collect::<Vec<_>>());
        }
        scored.into_iter().map(|(idx, _)| idx).collect()
    }
}

// --- Data Preprocessing ---
fn convert_token_list_to_scored_list(tokenized_string: Vec<usize>, idf_scores: &[f32]) -> Vec<(usize, f32)>{
    let token_counts = tokenized_string.iter().fold(HashMap::new(), |mut acc, &token| {
        *acc.entry(token as usize).or_insert(0) += 1;
        acc
    });
    let scored_list = token_counts.iter()
        .map(|(&token, &count)| {
            let idf = idf_scores.get(token).cloned().unwrap_or(0.0);
            (token, count as f32 * idf)
        })
        .collect::<Vec<(usize, f32)>>();


    let weights_sum = scored_list.iter().map(|&(_, weight)| weight).sum::<f32>();
    let normalized_weights: Vec<(usize, f32)> = scored_list.iter()
        .map(|&(index, weight)| (index, weight / weights_sum))
        .collect();
    normalized_weights
}

// --- Python Bindings ---
#[pyfunction]
fn build_approx_index(
    documents: Vec<Vec<usize>>,
    vocab_size: usize,
    idf_weights: Vec<f32>,
    micro: bool
) -> PyResult<Index> {
    println!("[al-rust] Building approximate index with {} documents, vocab_size={}, micro={}", documents.len(), vocab_size, micro);
    Ok(Index::build(documents, vocab_size, idf_weights, micro))
}

#[pyfunction]
fn search_approx_index(
    index: &Index,
    query: Vec<usize>,
    top_k: usize,
    micro: bool
) -> PyResult<Vec<usize>> {
    Ok(index.search(query, top_k, micro))
}

#[pymodule]
fn approximate_lexical(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_approx_index, m)?)?;
    m.add_function(wrap_pyfunction!(search_approx_index, m)?)?;
    Ok(())
}