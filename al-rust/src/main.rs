use rust_stemmers::{Algorithm, Stemmer};
use stopwords::{Spark, Language, Stopwords};
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use itertools::Itertools;
use std::time::Instant;
use serde_json;

macro_rules! profile {
    ($name:expr, $block:block) => {{
        let start = Instant::now();
        let result = $block;
        println!("{} took {:?}", $name, start.elapsed());
        result
    }};
}

// --- Utility Functions ---

fn get_progress_bar(total: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(msg);
    pb
}

fn tokenize_string(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

// --- Corpus Processing ---

struct CorpusProcessor {
    stemmer: Stemmer,
    stopwords: HashSet<String>,
}

impl CorpusProcessor {
    fn new() -> Self {
        let stemmer = Stemmer::create(Algorithm::English);
        let stopwords: HashSet<_> = stopwords::get(Language::English, Spark)
            .unwrap()
            .into_iter()
            .collect();
        Self {
            stemmer,
            stopwords,
        }
    }

    fn preprocess(&self, corpus: &[String]) -> Vec<Vec<String>> {
        profile!("Preprocessing corpus", {
            corpus
                .par_iter()
                .map(|doc| {
                    tokenize_string(doc)
                        .into_iter()
                        .filter(|token| !self.stopwords.contains(token))
                        .map(|token| self.stemmer.stem(&token))
                        .collect()
                })
                .collect()
        })
    }
}

// --- Tokenized Corpus ---

struct TokenizedCorpus {
    documents: Vec<Vec<usize>>,
    scored_documents: Vec<Vec<(usize, f32)>>,
    doc_token_counts: Vec<HashMap<usize, usize>>,
    doc_lengths: Vec<usize>,
    token_counts: Vec<usize>,
    word_to_index: HashMap<String, usize>,
}

impl TokenizedCorpus {
    fn from_corpus(
        corpus: &[String],
        k1: f32,
        b: f32,
        processor: &CorpusProcessor,
    ) -> Self {
        profile!("Tokenizing and scoring corpus", {
            let tokenized: Vec<Vec<String>> = processor.preprocess(corpus);

            // Build vocabulary
            let mut word_to_index = HashMap::new();
            for doc in &tokenized {
                for word in doc {
                    word_to_index.entry(word.clone()).or_insert(word_to_index.len());
                }
            }
            let vocab_size = word_to_index.len();

            // Tokenize documents to indices
            let documents: Vec<Vec<usize>> = tokenized
                .iter()
                .map(|doc| {
                    doc.iter()
                        .filter_map(|w| word_to_index.get(w).cloned())
                        .collect()
                })
                .collect();

            // Count tokens per document
            let doc_token_counts: Vec<HashMap<usize, usize>> = documents
                .iter()
                .map(|doc| {
                    let mut map = HashMap::new();
                    for &token in doc {
                        *map.entry(token).or_insert(0) += 1;
                    }
                    map
                })
                .collect();

            // Token counts across all docs
            let mut token_counts = vec![0; vocab_size];
            for doc in &doc_token_counts {
                for (&token, &count) in doc {
                    token_counts[token] += count;
                }
            }

            // Document lengths
            let doc_lengths: Vec<usize> = doc_token_counts
                .iter()
                .map(|doc| doc.values().sum())
                .collect();

            // Document frequency for each token
            let mut n_q = vec![0; vocab_size];
            for doc in &doc_token_counts {
                for (&token, _) in doc {
                    n_q[token] += 1;
                }
            }

            // IDF calculation
            let num_documents = documents.len() as f32;
            let idf: Vec<f32> = n_q
                .iter()
                .map(|&df| ((num_documents - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln())
                .collect();

            // BM25 scoring
            let avg_doc_length = doc_lengths.iter().sum::<usize>() as f32 / num_documents;
            let scored_documents: Vec<Vec<(usize, f32)>> = doc_token_counts
                .iter()
                .map(|doc| {
                    doc.iter()
                        .map(|(&token, &count)| {
                            let dl = doc_lengths[doc_token_counts.iter().position(|d| d == doc).unwrap()] as f32;
                            let tf = count as f32;
                            let denom = tf + k1 * (1.0 - b + b * (dl / avg_doc_length));
                            let score = idf[token] * tf * (k1 + 1.0) / denom;
                            (token, score)
                        })
                        .collect()
                })
                .collect();

            Self {
                documents,
                scored_documents,
                doc_token_counts,
                doc_lengths,
                token_counts,
                word_to_index,
            }
        })
    }
}

// --- Hashing and Indexing ---

fn hash_vector(vector: &[(usize, f32)], hash_proj: &[Vec<i8>]) -> u16 {
    // Simple locality-sensitive hash: sum of projections
    let mut hash: u16 = 0;
    for (i, proj) in hash_proj.iter().enumerate() {
        let mut score = 0.0;
        for &(idx, val) in vector {
            score += proj[idx % proj.len()] as f32 * val;
        }
        if score > 0.0 {
            hash |= 1 << i;
        }
    }
    hash
}

struct TermCluster {
    hash_proj: Vec<Vec<i8>>,
    hashes_to_documents: HashMap<u16, Vec<usize>>,
}

impl TermCluster {
    fn new(documents: &[Vec<(usize, f32)>], hash_dim: usize, vocab_size: usize) -> Self {
        profile!("Building TermCluster", {
            let mut rng = rand::thread_rng();
            let hash_proj: Vec<Vec<i8>> = (0..hash_dim)
                .map(|_| {
                    (0..vocab_size)
                        .map(|_| if rng.gen_bool(0.5) { -1 } else { 1 })
                        .collect()
                })
                .collect();

            let mut hashes_to_documents: HashMap<u16, Vec<usize>> = HashMap::new();
            for (doc_idx, doc) in documents.iter().enumerate() {
                let hash = hash_vector(doc, &hash_proj);
                hashes_to_documents.entry(hash).or_default().push(doc_idx);
            }

            Self {
                hash_proj,
                hashes_to_documents,
            }
        })
    }

    fn search(&self, query: &[(usize, f32)], top_k: usize) -> Vec<usize> {
        profile!("TermCluster search", {
            let query_hash = hash_vector(query, &self.hash_proj);
            // BFS Hamming search
            let mut results = Vec::new();
            for dist in 0..=16 {
                for hash in (0..=u16::MAX).filter(|h| hamming_distance(*h, query_hash) == dist) {
                    if let Some(docs) = self.hashes_to_documents.get(&hash) {
                        results.extend(docs);
                        if results.len() >= top_k {
                            return results.into_iter().take(top_k).collect();
                        }
                    }
                }
            }
            results.into_iter().take(top_k).collect()
        })
    }
}

fn hamming_distance(a: u16, b: u16) -> u32 {
    (a ^ b).count_ones()
}

// --- Index ---

struct Index {
    posting_list: Vec<TermCluster>,
    doc_list: Vec<Vec<(usize, f32)>>,
    word_to_index: HashMap<String, usize>,
}

impl Index {
    fn build(tokenized: &TokenizedCorpus, hash_dim: usize) -> Self {
        profile!("Building Index", {
            let posting_list: Vec<TermCluster> = (0..tokenized.token_counts.len())
                .map(|_| {
                    TermCluster::new(
                        &tokenized.scored_documents,
                        hash_dim,
                        tokenized.token_counts.len(),
                    )
                })
                .collect();

            Self {
                posting_list,
                doc_list: tokenized.scored_documents.clone(),
                word_to_index: tokenized.word_to_index.clone(),
            }
        })
    }

    fn search(
        &self,
        query: &[(usize, usize)],
        top_k: usize,
    ) -> Vec<usize> {
        profile!("Index search", {
            // Convert query to BM25-style vector (with dummy scores)
            let query_vec: Vec<(usize, f32)> = query.iter().map(|&(idx, count)| (idx, count as f32)).collect();
            let mut candidates = HashSet::new();
            for (token_idx, _) in query {
                let docs = self.posting_list[*token_idx].search(&query_vec, top_k);
                candidates.extend(docs);
            }

            // Score candidates
            let mut scored: Vec<(usize, f32)> = candidates
                .into_iter()
                .map(|doc_idx| {
                    let doc_vec = &self.doc_list[doc_idx];
                    let mut score = 0.0;
                    let mut dv_idx = 0;
                    let mut q_idx = 0;
                    let mut query_sorted = query_vec.clone();
                    query_sorted.sort_by_key(|x| x.0);
                    let mut doc_sorted = doc_vec.clone();
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
                    (doc_idx, score)
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(top_k);
            scored.into_iter().map(|(idx, _)| idx).collect()
        })
    }
}

// --- PyO3 Bindings ---

#[pyfunction]
fn build_approx_index(documents: Vec<String>) -> PyResult<Index> {
    let processor = CorpusProcessor::new();
    let tokenized = TokenizedCorpus::from_corpus(&documents, 1.5, 0.75, &processor);
    Ok(Index::build(&tokenized, 16))
}

#[pyfunction]
fn search_approx_index(
    index: &Index,
    query: Vec<(usize, usize)>,
    top_k: usize,
) -> PyResult<Vec<usize>> {
    Ok(index.search(&query, top_k))
}

#[pyfunction]
fn tokenize_query(query: String, tokenization: HashMap<String, usize>) -> PyResult<Vec<(usize, usize)>> {
    let processor = CorpusProcessor::new();
    let tokens: Vec<String> = tokenize_string(&query);
    let filtered: Vec<String> = tokens
        .into_iter()
        .filter(|token| !processor.stopwords.contains(token))
        .map(|token| processor.stemmer.stem(&token))
        .collect();

    let mut counts = HashMap::new();
    for token in filtered {
        if let Some(&idx) = tokenization.get(&token) {
            *counts.entry(idx).or_insert(0) += 1;
        }
    }
    Ok(counts.into_iter().collect())
}

#[pymodule]
fn al_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_approx_index, m)?)?;
    m.add_function(wrap_pyfunction!(search_approx_index, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_query, m)?)?;
    Ok(())
}

fn main() {
    println!("Approximate Lexical Rust Module");
}