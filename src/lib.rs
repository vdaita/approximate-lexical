mod weighted_lsh_searcher;
mod random_proj;
mod utils;
mod compression;
mod sample_searcher;

use std::collections::{HashMap};
use pyo3::prelude::*;
use rand_distr::{Distribution};
use crate::weighted_lsh_searcher::WeightedMinHashLSH;
use crate::compression::compress_vector_alpha;
use crate::sample_searcher::SampleSearcher;

// we want the query to be weighted by the IDF scores
// we want the documents to have the rest of the BM25 formula

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
    idf_weights: Vec<f32>,

    k1: f32,
    b: f32,
    compression_alpha: f32 // set to 1.0 to disable compression
}

impl Index {
    fn build(tokenized_documents: Vec<Vec<usize>>, vocab_size: usize, idf_weights: Vec<f32>, k1: f32, b: f32, compression_alpha: f32, micro: bool) -> Self {
        if micro {
            println!("[micro] Index::build called");
        }

        let average_doc_length = (tokenized_documents.iter().map(|doc| doc.len()).sum::<usize>() as f32) / (tokenized_documents.len() as f32);

        let scored_documents: Vec<Vec<(usize, f32)>> = tokenized_documents
            .into_iter()
            .map(|tokens| convert_document_list_to_scored_list(tokens, k1, b, average_doc_length))
            .collect();
        let scored_documents = scored_documents
            .into_iter()
            .map(|doc| compress_vector_alpha(&doc, compression_alpha))
            .collect::<Vec<Vec<(usize, f32)>>>();
        
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
            k1,
            b,
            compression_alpha
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
        let query_vec = convert_query_list_to_scored_list(tokenized_query, &self.idf_weights);
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
fn convert_document_list_to_scored_list(tokenized_string: Vec<usize>, k1: f32, b: f32, average_doc_length: f32) -> Vec<(usize, f32)>{
    let token_counts = tokenized_string.iter().fold(HashMap::new(), |mut acc, &token| {
        *acc.entry(token as usize).or_insert(0) += 1;
        acc
    });
    let total_tokens = tokenized_string.len();
    let scored_list = token_counts.iter()
        .map(|(&token, &count)| {
            let score = ((count as f32) * (k1 + 1.0)) / ((count as f32) + k1 * (1.0 - b + b * ((total_tokens as f32) / (average_doc_length as f32))));
            (token, score)
        })
        .collect::<Vec<(usize, f32)>>();

    scored_list
}

fn convert_query_list_to_scored_list(tokenized_string: Vec<usize>, idf_scores: &[f32]) -> Vec<(usize, f32)> {
    let token_counts = tokenized_string.iter().fold(HashMap::new(), |mut acc, &token| {
        *acc.entry(token as usize).or_insert(0) += 1;
        acc
    });
    let count_list = token_counts.iter()
        .map(|(&token, &count)| (token, idf_scores[token] * count as f32))
        .collect::<Vec<(usize, f32)>>();
    count_list
}

// --- Python Bindings ---
#[pyfunction]
fn build_approx_index(
    documents: Vec<Vec<usize>>,
    vocab_size: usize,
    idf_weights: Vec<f32>,
    k1: f32,
    b: f32,
    compression_alpha: f32,
    micro: bool
) -> PyResult<Index> {
    println!(
        "[al-rust] Building approximate index with {} documents, vocab_size={}, k1={}, b={}, compression_alpha={}, micro={}",
        documents.len(),
        vocab_size,
        k1,
        b,
        compression_alpha,
        micro
    );
    Ok(Index::build(documents, vocab_size, idf_weights, k1, b, compression_alpha, micro))
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