use std::collections::{HashMap};
use pyo3::prelude::*;
use rand::rng;
use rand_distr::{Distribution, StandardNormal};
// --- Hashing and Indexing ---

fn hash_vector(vector: &[(usize, f32)], hash_proj: &[Vec<f32>]) -> u16 {
    let mut hash: u16 = 0;
    for (i, proj) in hash_proj.iter().enumerate() {
        let mut score = 0.0;
        for &(idx, val) in vector {
            score += proj[idx % proj.len()] * val;
        }
        if score > 0.0 {
            hash |= 1 << i;
        }
    }
    hash
}

#[pyclass]
#[derive(Clone)]
struct TermCluster {
    hash_proj: Vec<Vec<f32>>,
    hashes_to_documents: HashMap<u16, Vec<usize>>,
}

impl TermCluster {
    fn new(documents: &[Vec<(usize, f32)>], vocab_size: usize, micro: bool) -> Self {
        if micro {
            println!("[micro] TermCluster::new called with {} documents, vocab_size={}", documents.len(), vocab_size);
        }
        let hash_dim = 32;
        let mut rng = rng();
        let hash_proj: Vec<Vec<f32>> = (0..hash_dim)
            .map(|_| {
                (0..vocab_size)
                    .map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng))
                    .collect()
            })
            .collect();

        let mut hashes_to_documents: HashMap<u16, Vec<usize>> = HashMap::new();
        for (doc_idx, doc) in documents.iter().enumerate() {
            let hash = hash_vector(doc, &hash_proj);
            hashes_to_documents.entry(hash).or_default().push(doc_idx);
            if micro && doc_idx < 5 {
                println!("[micro] Document {} hashed to {}", doc_idx, hash);
            }
        }

        Self {
            hash_proj,
            hashes_to_documents,
        }
    }

    fn search(&self, query: &[(usize, f32)], top_k: usize, micro: bool) -> Vec<usize> {
        if micro {
            println!("[micro] TermCluster::search called with top_k={}", top_k);
        }
        let query_hash = hash_vector(query, &self.hash_proj);
        if micro {
            println!("[micro] Query hash: {}", query_hash);
        }
        // BFS Hamming search
        let mut results = Vec::new();
        for dist in 0..=16 {
            if micro {
                println!("[micro] Hamming distance: {}", dist);
            }
            let mut found = 0;
            for hash in (0..=u16::MAX).filter(|h| hamming_distance(*h, query_hash) == dist) {
                if let Some(docs) = self.hashes_to_documents.get(&hash) {
                    results.extend(docs);
                    found += docs.len();
                    if micro {
                        println!("[micro] Found {} docs at hash {} (dist {})", docs.len(), hash, dist);
                    }
                    if results.len() >= top_k {
                        if micro {
                            println!("[micro] Enough results found, returning.");
                        }
                        return results.into_iter().take(top_k).collect();
                    }
                }
            }
            if micro {
                println!("[micro] Total found at dist {}: {}", dist, found);
            }
        }
        if micro {
            println!("[micro] Finished search, returning {} results.", results.len());
        }
        results.into_iter().take(top_k).collect()
    }
}

fn hamming_distance(a: u16, b: u16) -> u32 {
    (a ^ b).count_ones()
}

// --- Helper Functions ---

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
    clusters: TermCluster,
    doc_list: Vec<Vec<(usize, f32)>>,
    vocab_size: usize,
}

impl Index {
    fn build(scored_documents: &[Vec<(usize, f32)>], vocab_size: usize, micro: bool) -> Self {
        if micro {
            println!("[micro] Index::build called");
        }
        let clusters = TermCluster::new(scored_documents, vocab_size, micro);
        Self {
            clusters,
            doc_list: scored_documents.to_vec(),
            vocab_size,
        }
    }

    fn search(
        &self,
        query_vec: &[(usize, f32)],
        top_k: usize,
        micro: bool,
    ) -> Vec<usize> {
        if micro {
            println!("[micro] Index::search called");
        }
        let candidates = self.clusters.search(query_vec, top_k, micro); // Overfetch for better recall
        let mut scored: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|doc_idx| {
                let doc_vec = &self.doc_list[doc_idx];
                let score = sparse_inner_product(query_vec, doc_vec);
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

// --- Python Bindings ---

#[pyfunction]
fn build_approx_index(
    scored_documents: Vec<Vec<(usize, f32)>>,
    vocab_size: usize,
    micro: bool
) -> PyResult<Index> {
    println!("[al-rust] Building approximate index with {} documents, vocab_size={}, micro={}", scored_documents.len(), vocab_size, micro);
    Ok(Index::build(&scored_documents, vocab_size, micro))
}

#[pyfunction]
fn search_approx_index(
    index: &Index,
    query: Vec<(usize, f32)>,
    top_k: usize,
    micro: bool
) -> PyResult<Vec<usize>> {
    Ok(index.search(&query, top_k, micro))
}

#[pymodule]
fn al_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_approx_index, m)?)?;
    m.add_function(wrap_pyfunction!(search_approx_index, m)?)?;
    Ok(())
}