// al-rust/src/main.rs
use rust_stemmers::{Algorithm, Stemmer};
use stopwords::{Spark, Language, Stopwords};
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::prelude::*; // for python bindings
use kmeans::*;

const DIM_SIZE: i32 = 512; // Dimension size for the dense vectors

struct Block {
    summary_vector: Vec<(usize, f32)>, // Summary vector with (token, weight)
    content_vector: Vec<usize> // Indices of the content in the summary
}

struct TokenizedCorpus {
    documents: Vec<Vec<Vec<usize>>>,
    scored_documents: Vec<Vec<Vec<(usize, f32)>>>,
    doc_token_counts: Vec<HashMap<usize, usize>>,
    token_counts: Vec<usize>,
    word_to_index: HashMap<String, usize>
}

struct KMeansResult { 
    centroids: Vec<Vec<f32>>, // centroids of the clusters
    labels: Vec<usize> // labels for each document regarding which centroid they belong to
}

struct Index {
    posting_list: Vec<Vec<Block>>, // positing list for each token
    quantized_doc_list: Vec<Vec<(i32, u8)>>, // list of documents with their token counts / scores (separately)
    word_to_index: HashMap<String, usize>
}

fn get_progress_bar(total: u64) -> ProgressBar {
    let progress_bar = ProgressBar::new(total);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} docs")
            .unwrap()
            .progress_chars("#>-"),
    );
    progress_bar
}

fn tokenize_string(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn preprocess_corpus(corpus: &[&str]) -> Vec<Vec<String>> {
    let stemmer = Stemmer::create(Algorithm::English);
    let stopwords: HashSet<_> = stopwords::get(Language::English, Spark)
        .unwrap()
        .into_iter()
        .collect();
    corpus
        .par_iter()
        .map(|doc| {
            let tokens: Vec<String> = tokenize_string(doc);
            tokens
                .into_iter()
                .filter(|token| !stopwords.contains(token))
                .map(|token| stemmer.stem(&token))
                .collect()
        })
        .collect()
}

fn get_token_count_pairs_from_vec(list: Vec<usize>) -> Vec<(usize, usize)> {
    let mut token_count_map: HashMap<usize, usize> = HashMap::new();
    for &token in &list {
        *token_count_map.entry(token).or_insert(0) += 1;
    }
    token_count_map.into_iter().collect()
}

fn quantize_scores(scores: Vec<Vec<(usize, f32)>>) -> Vec<Vec<(usize, u8)>> {
    let all_scores = scores
        .iter()
        .flat_map(|doc| doc.iter().map(|(_, score)| *score))
        .collect::<Vec<f32>>()
        .sort();

    let bin_starts: Vec<f32> = (0..256)
        .map(|i| all_scores[i * all_scores.len() / 256])
        .collect();

    scores
        .iter()
        .map(|doc_score| {
            doc_score
                .iter()
                .map(|&(token, score)| {
                    let quantized_score = bin_starts
                        .iter()
                        .position(|&start| score < start)
                        .unwrap_or(255) as u8; // 255 is the max bin index
                    (token, quantized_score)
                })
                .collect::<Vec<(usize, u8)>>()
        })
        .collect::<Vec<Vec<(usize, u8)>>>()
}

fn tokenize_corpus(corpus: &[&str], k1: f32, b: f32) -> TokenizedCorpus {
    println!("Tokenizing corpus...");
    let tokenized_corpus: Vec<Vec<String>> = preprocess_corpus(corpus);
    let mut word_to_index = HashMap::new();
    let mut doc_token_counts: Vec<HashMap<usize, usize>> = vec![HashMap::new(); tokenized_corpus.len()];
    let mut token_counts: Vec<usize> = vec![0; word_to_index.len()];
    let mut n_q: Vec<usize> = vec![0; word_to_index.len()];
    let mut idf: Vec<f32> = vec![0.0; word_to_index.len()];

    // create a mapping from words to indices
    let progress_bar = get_progress_bar(tokenized_corpus.len() as u64);

    for (i, doc) in tokenized_corpus.iter().enumerate() {
        for word in doc {
            if !word_to_index.contains_key(word) {
                word_to_index.insert(word.clone(), word_to_index.len());
            }
        }
        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Tokenization complete");
    progress_bar.reset();

    // keep track of token counts in each document
    for (i, doc) in tokenized_corpus.iter().enumerate() {
        for word in doc {
            let word_index = word_to_index[word];
            *doc_token_counts[i].entry(word_index).or_insert(0) += 1;
            token_counts[word_index] += 1;
        }
        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Token counts complete");
    progress_bar.reset();

    for (doc_i, doc) in doc_token_counts.iter().enumerate() {
        for (&token_index, _) in doc.iter() {
            n_q[token_index] += 1; // just save once for the number of documents that contains the token
        }
    }

    progress_bar.finish_with_message("Document token counts complete");
    progress_bar.reset();

    let num_documents = tokenized_corpus.len() as i32;

    progress_bar.set_length(token_counts.len() as u64);
    progress_bar.set_message("Calculating IDF...");
    for (tok_i, tok_count) in token_counts.iter().enumerate() {
        let idf_i = ln(((num_documents - n_q[tok_i] + 0.5) as f32 / (n_q[tok_i] + 0.5) as f32) + 1);
        idf[tok_i] = idf_i;
        progress_bar.inc(1);
    }
    progress_bar.finish_with_message("IDF calculation complete");

    progress_bar.reset();
    progress_bar.set_length(corpus.len() as u64);
    progress_bar.set_message("Tokenizing corpus...");
    corpus
        .par_iter()
        .map(|doc| {
            let tokens: Vec<String> = tokenize_string(doc);
            progress_bar.inc(1);
            tokens
                .into_iter()
                .filter_map(|token| word_to_index.get(&token).cloned())
                .collect();
        })
        .collect();
    progress_bar.finish_with_message("Corpus tokenization complete");

    // Convert tokenized corpus from Vec<Vec<Vec<usize>>> to Vec<Vec<Vec<usize, usize>>>
    let doc_token_count_pairs: Vec<Vec<Vec<(usize, usize)>>> = tokenized_corpus
        .into_iter()
        .map(|doc| {
            get_token_count_pairs_from_vector(doc.into_iter())
        });

    let avg_doc_length: f32 = token_counts.sum() as f32 / num_documents as f32;

    let doc_score_pairs: Vec<Vec<Vec<(usize, f32)>>> = doc_token_count_pairs
        .into_iter()
        .enumerate()
        .map(|(doc_index, doc)| {
            doc.into_iter()
                .map(|doc_token, doc_count| {
                    let idf_qi = doc_count * (k1 + 1) / (doc_count * (k1 * (1 - b + b * (token_counts[doc_token] as f32 / avg_doc_length))));
                    let score = idf_qi * idf[doc_token];
                    (doc_token, score)
                });
        });

    let all_doc_scores: Vec<f32> = doc_score_pairs
        .iter()
        .flat_map(|doc| doc.iter().map(|(_, score)| *score))
        .collect();


    let quantized_doc_score_pairs: Vec<Vec<Vec<i32, i8>>> = doc_score_pairs
        .into_iter()
        .map(|doc| {
            doc.into_iter()
                .map(|(token, score)| (token as i32, (score * 1000.0) as i16)) // quantize the score to i16
                .collect();
        });

    TokenizedCorpus {
        documents: tokenized_corpus,
        doc_token_counts,
        token_counts,
        word_to_index
    }
}

// strategy: first, build a series of clusters for a very small number of documents, and take the max for each coordinate
// next, for the next level of centroids, take the average from the closest clusters
// outputs: centroids (sparse summary vector) and their corresponding documents (Vec<Block>)

#[pyfunction]
fn build_approx_index(documents: &[Vec<String>]) -> Vec<Vec<Block>>{
    println!("Building index...");
    let tokenized_corpus: TokenizedCorpus = tokenize_corpus(documents);
    let mut index: Vec<Vec<Block>> = vec![Vec::new(); tokenized_corpus.word_to_index.len()];

    let progress_bar = get_progress_bar(tokenized_corpus.documents.len() as u64);
    let mut raw_posting_lists: Vec<Vec<Vec<(usize, f32)>>> = vec![Vec::new(); tokenized_corpus.word_to_index.len()];

    // Separate the documents into posting lists
    for (doc_id, doc) in tokenized_corpus.documents.iter().enumerate() {
        let mut doc_vector: Vec<(usize, f32)> = Vec::new();
        for &token_index in doc {
            let count = tokenized_corpus.doc_token_counts[doc_id].get(&token_index).cloned().unwrap_or(0);
            if count > 0 {
                doc_vector.push((token_index, count as f32));
            }
        }
        for &(token_index, weight) in &doc_vector {
            raw_posting_lists[token_index].push(doc_vector.clone());
        }
        progress_bar.inc(1);
    }

    // Compute centroids and assign documents to their randomly assigned centroids based on distance
    let mut centroids: Vec<Vec<Vec<(usize, f32)>>> = vec![Vec::new(); tokenized_corpus.word_to_index.len()];
    for (token_index, posting_list) in raw_posting_lists.iter().enumerate() {

    }
}

#[pyfunction]
fn search_approx_index(posting_list: Vec<Vec<Block>>, doc_list: Vec<Vec<(i32, usize)>>, query: Vec<(usize, usize)>, top_k: i32, num_blocks_per_token_per_k: i32) -> Vec<usize> { // there should be a pair of i32, usize which represents the count
    let first_pass_per_token = top_k * num_blocks_per_token_per_k;
    let mut top_summary_vectors: Vec<Vec<usize>> = query
        .into_iter()
        .map(|(token_id, token_count)| {
            let mut block_scores: Vec<f32> = Vec::new();
            for block in &posting_list[token_id] {
                let mut block_score = 0.0;
                
                let mut query_index: i32 = 0;
                let mut summary_index: i32 = 0;
                while (query_index < query.len() && summary_index < block.summary_vector.len()) {
                    if query[query_index].0 == block.summary_vector[summary_index].0 {
                        block_score += block.summary_vector[summary_index].1 * token_count as f32;
                        query_index += 1;
                    } else if query[query_index].0 < block.summary_vector[summary_index].0 {
                        query_index += 1;
                    } else {
                        summary_index += 1;
                    }
                }

                block_scores.push(block_score);
            }

            let mut scored_blocks: Vec<(usize, f32)> = block_scores
                .into_iter()
                .enumerate()
                .collect();
            scored_blocks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored_blocks.truncate(top_k * num_blocks_per_token_per_k as usize);
            return scored_blocks.map(|a, b| a)
        }); 
    
    for (token_id, token_count) in query {
        println!("Analyzing the first {} blocks out of {} potential blocks for token {}", first_pass_per_token, posting_list[token_id].len(), token_id);
    }

    let dedup_documents: Vec<usize> = top_summary_vectors
        .into_iter()
        .flat_map(|blocks| doc_list[block])
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    
    // for each of the blocks, we need to identify the unique documents, and then calculate the scores
    let scored_documents = dedup_documents
        .into_iter()
        .map(|doc_id| {
            let mut score = 0.0;
            let mut doc_i = 0;
            let mut query_i = 0;
            while doc_i < top_summary_vectors.len() && query_i < query.len() {
                if top_summary_vectors[doc_i].0 == query[query_i].0 {
                    score += top_summary_vectors[doc_i].1 * query[query_i].1 as f32;
                    doc_i += 1;
                    query_i += 1;
                } else if top_summary_vectors[doc_i].0 < query[query_i].0 {
                    doc_i += 1;
                } else {
                    query_i += 1;
                }
            }
            (doc_id, score)
        })
        .collect::<Vec<_>>();

    // These documents were properly scored
    scored_documents
}

#[pyfunction]
fn tokenize_query(query: String, tokenization: HashMap<String, usize>) -> Vec<(usize, usize)> {
    let stemmer = Stemmer::create(Algorithm::English);
    let stopwords: HashSet<_> = stopwords::get(Language::English, Spark)
        .unwrap()
        .into_iter()
        .collect();
    
    let tokens: Vec<String> = tokenize_string(&query);
    tokens
        .into_iter()
        .filter(|token| !stopwords.contains(token))
        .map(|token| stemmer.stem(&token))
        .filter_map(|token| tokenization.get(&token).cloned())
        .collect()
}

#[pymodule]
fn al_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_approx_index, m)?)?;
    m.add_function(wrap_pyfunction!(search_approx_index, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_query, m)?)?;
    Ok(())
}

fn main() {
    println!("Approximate Lexical Rust Module");
}
