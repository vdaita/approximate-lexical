// al-rust/src/main.rs
use rust_stemmers::{Algorithm, Stemmer};
use stopwords::{Spark, Language, Stopwords};
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

struct Block {
    summary_vector: Vec<(usize, f32)>, // Summary vector with (token, weight)
    content_vector: Vec<usize> // Indices of the content in the summary
}

struct TokenizedCorpus {
    documents: Vec<Vec<Vec<usize>>>,
    doc_token_counts: Vec<HashMap<usize, usize>>,
    token_counts: Vec<usize>,
    word_to_index: HashMap<String, usize>
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


fn tokenize_corpus(corpus: &[&str]) -> TokenizedCorpus {
    println!("Tokenizing corpus...");
    let tokenized_corpus: Vec<Vec<String>> = preprocess_corpus(corpus);
    let mut word_to_index = HashMap::new();
    let mut doc_token_counts: Vec<HashMap<usize, usize>> = vec![HashMap::new(); tokenized_corpus.len()];
    let mut token_counts: Vec<usize> = vec![0; word_to_index.len()];


    // create a mapping from words to indices
    let progress_bar = ProgressBar::new(tokenized_corpus.len() as u64);
    progress_bar.set_style(
    ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} docs")
        .unwrap()
        .progress_chars("#>-"),
    );

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
    
    TokenizedCorpus {
        documents: tokenized_corpus,
        doc_token_counts,
        token_counts,
        word_to_index
    }

}

fn build_approx_index(tokenized_documents: &[Vec<Vec<usize>>]) -> Vec<Vec<Block>>{
    println!("Building index...");
    let mut index: Vec<Vec<Block>> = Default::default();

}

fn search_approx_index(tokenized_query: &[Vec<usize>]) -> Vec<usize> {
    println!("Searching index...");

    println!("Checking the posting list for each token in the query");

    println!("For each token, selecting the top-k blocks based on the summary vector weights");

    println!("Merging the results from the top-k blocks to form the final result set");
}



fn main() {
    println!("Hello, world!");
}
