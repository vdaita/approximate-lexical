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

// TODO: finish computing the scores for the tokens in this function
fn tokenize_corpus(corpus: &[&str]) -> TokenizedCorpus {
    println!("Tokenizing corpus...");
    let tokenized_corpus: Vec<Vec<String>> = preprocess_corpus(corpus);
    let mut word_to_index = HashMap::new();
    let mut doc_token_counts: Vec<HashMap<usize, usize>> = vec![HashMap::new(); tokenized_corpus.len()];
    let mut token_counts: Vec<usize> = vec![0; word_to_index.len()];


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

    let doc_score_pairs: Vec<Vec<Vec<(usize, f32)>>> = doc_token_count_pairs
        .into_iter()
        .enumerate()
        .map(|(doc_index, doc)| {
            doc.into_iter()
                .map(|token, count| {
                    let weight = token_counts[token] as f32 / token_counts.len() as f32;

                });
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
}

#[pyfunction]
fn search_approx_index(posting_list: Vec<Vec<Block>>, doc_list: Vec<Vec<(i32, usize)>>, query: Vec<(usize, usize)>, top_k: i32, num_blocks_per_token_per_k: i32) -> Vec<usize> { // there should be a pair of i32, usize which represents the count
    let first_pass_per_token = top_k * num_blocks_per_token_per_k;
    let mut top_summary_vectors = query
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
            scored_blocks
        }); 
    
    for (token_id, token_count) in query {
        print("Analyzing the first {} blocks out of {} potential blocks for token {}");
    }
    
    // for each of the blocks, we need to identify the unique documents, and then calculate the scores

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



fn main() {
    println!("Hello, world!");
}
