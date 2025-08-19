use bm25::{
    DefaultTokenizer, Document, EmbedderBuilder, Embedding, Language, ScoredDocument, Scorer,
    SearchEngineBuilder, SearchResult,
};
use clap::Parser;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

mod global_index;
mod kmeans;
mod sparse_to_dense;
mod term_index;
mod utils;

use crate::global_index::GlobalIndex;
use crate::utils::ApproximateLexicalParameters;
use log::{trace, debug, info};
use env_logger;

#[derive(Deserialize)]
struct DataSet {
    documents: Vec<String>,
    queries: Vec<String>,
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long, default_value_t=String::from("scifact"))]
    dataset_source: String,

    #[arg(short, long, default_value_t = 64)]
    top_k: usize,

    #[arg(short, long, default_value_t = 64)]
    clusters_searched: usize,

    #[arg(short, long, default_value_t = 0)]
    queries: usize,

    #[arg(long, default_value_t = 0)] // If the value is zero, then all documents work
    documents: usize,

    #[arg(long, default_value_t = 5)]
    max_top_k_multiple: usize,
}

fn main() {
    env_logger::init();
    info!("Approximate BM25 Search");
    
    let args = Args::parse();
    let dataset_path = format!("data/{}.json", args.dataset_source);
    let file = File::open(dataset_path.clone()).expect("Unable to open dataset file");
    let reader = BufReader::new(file);

    let dataset: DataSet = serde_json::from_reader(reader).expect("JSON was not well-formatted");
    let mut documents: Vec<String> = dataset.documents;
    let mut queries: Vec<String> = dataset.queries;

    println!(
        "Read {} documents and {} queries from {}",
        documents.len(),
        queries.len(),
        dataset_path.clone()
    );

    if args.documents > 0 {
        documents = documents.iter().take(args.documents).cloned().collect();
        println!("Using {} documents for the search.", documents.len());
    }

    if args.queries > 0 {
        queries = queries.iter().take(args.queries).cloned().collect();
        println!("Using {} queries for the search.", queries.len());
    }

    let document_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
    let queries_refs: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();

    let tokenizer = DefaultTokenizer::builder()
        .language_mode(Language::English)
        .normalization(true)
        .stopwords(true)
        .stemming(true)
        .build();

    let embedder =
        EmbedderBuilder::<u32>::with_tokenizer_and_fit_to_corpus(tokenizer, &document_refs).build();

    let embedded_documents = document_refs
        .iter()
        .map(|doc| embedder.embed(doc))
        .collect::<Vec<_>>();
    let embedded_queries = queries_refs
        .iter()
        .map(|query| embedder.embed(query))
        .collect::<Vec<_>>();

    let mut scorer = Scorer::<usize>::new();
    for (i, embedded_document) in embedded_documents.iter().enumerate() {
        scorer.upsert(&i, embedded_document.clone());
    }

    // Build approximate search engine with default parameters
    let approx_params = ApproximateLexicalParameters::new(
        16,    // cluster_size
        16,    // dense_dim_size
        100,   // kmeanspp_sample
        10,    // kmeans_iterations
        1000,  // kmeans_batch_size
        false, // spherical
        1,     // num_cluster_segments
        0.9,   // alpha_significance_threshold
    );

    let embedded_docs_sparse: Vec<Vec<(usize, f32)>> = embedded_documents
        .iter()
        .map(|embedding| {
            embedding
                .iter()
                .map(|embed| (embed.index as usize, embed.value))
                .collect()
        })
        .collect();

    let approximate_search_engine = GlobalIndex::new(embedded_docs_sparse, approx_params);

    // Performance tracking
    let mut regular_times = Vec::new();
    let mut approx_times = Vec::new();
    let mut recalls = Vec::new();
    let mut precisions = Vec::new();

    // Different values of m (multiple of k)
    let m_values: Vec<usize> = (0..args.max_top_k_multiple).into_iter().collect();

    for (query_idx, query_embedding) in embedded_queries.iter().enumerate() {
        let approx_query_embedding = query_embedding
            .iter()
            .map(|embed| (embed.index as usize, embed.value))
            .collect::<Vec<(usize, f32)>>();
        
        let regular_start_time = Instant::now();
        let mut original_results: Vec<ScoredDocument<usize>> =
            scorer.matches(&query_embedding).into_iter().collect();
        original_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let original_results: Vec<ScoredDocument<usize>> =
            original_results.into_iter().take(args.top_k).collect();
        let regular_end_time = Instant::now();

        for &m in &m_values {
            let extended_top_k = args.top_k * m;

            let approx_start_time = Instant::now();
            let approx_results: Vec<(usize, f32)> = approximate_search_engine.query(
                &approx_query_embedding,
                args.clusters_searched, // default # clusters
                extended_top_k,
            );
            let approx_end_time = Instant::now();

            // sort and get top-k
            let mut approx_results_sorted = approx_results;
            approx_results_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let approx_top_k: Vec<u32> = approx_results_sorted
                .into_iter()
                .take(args.top_k)
                .map(|(doc_id, _)| doc_id as u32)
                .collect();

            let original_doc_ids: std::collections::HashSet<u32> = original_results
                .iter()
                .map(|result| result.id as u32)
                .collect();
            let approx_doc_ids: std::collections::HashSet<u32> =
                approx_top_k.iter().cloned().collect();

            println!(
                "Retrieved {} approximate documents and {} regular documents.",
                approx_doc_ids.len(),
                original_doc_ids.len()
            );

            let intersection_size = original_doc_ids.intersection(&approx_doc_ids).count();
            let recall = intersection_size as f64 / original_doc_ids.len() as f64;
            let precision = if approx_doc_ids.len() > 0 {
                intersection_size as f64 / approx_doc_ids.len() as f64
            } else {
                0.0
            };

            let regular_time = (regular_end_time - regular_start_time).as_micros();
            let approx_time = (approx_end_time - approx_start_time).as_micros();

            println!(
                "Query {}: m={}, Regular: {}μs, Approx: {}μs, Recall: {:.4}, Precision: {:.4}",
                query_idx + 1,
                m,
                regular_time,
                approx_time,
                recall,
                precision
            );

            // Store metrics for the base case (m=1) for aggregate analysis
            if m == 1 {
                regular_times.push(regular_time);
                approx_times.push(approx_time);
                recalls.push(recall);
                precisions.push(precision);
            }
        }
    }

    // Print aggregate analysis
    let avg_regular_time = regular_times.iter().sum::<u128>() as f64 / regular_times.len() as f64;
    let avg_approx_time = approx_times.iter().sum::<u128>() as f64 / approx_times.len() as f64;
    let avg_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
    let avg_precision = precisions.iter().sum::<f64>() / precisions.len() as f64;
    let speedup = avg_regular_time / avg_approx_time;

    println!("\n=== Aggregate Results ===");
    println!("Dataset: {}", args.dataset_source);
    println!("Top-k: {}", args.top_k);
    println!("Number of queries: {}", queries.len());
    println!("Number of documents: {}", documents.len());
    println!();
    println!("Average regular search time: {:.2}μs", avg_regular_time);
    println!("Average approximate search time: {:.2}μs", avg_approx_time);
    println!("Average speedup: {:.2}x", speedup);
    println!("Average recall: {:.4}", avg_recall);
    println!("Average precision: {:.4}", avg_precision);
    println!(
        "Average F1-score: {:.4}",
        2.0 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    );
}
