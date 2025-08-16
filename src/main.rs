use bm25::{EmbedderBuilder, Language, SearchEngineBuilder, SearchResult};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize)]
struct DataSet {
    documents: Vec<String>,
    queries: Vec<String>,
}

fn main() {
    let dataset_path = "data/scifact.json";
    let file = File::open(dataset_path).expect("Unable to open dataset file");
    let reader = BufReader::new(file);

    let dataset: DataSet = serde_json::from_reader(reader).expect("JSON was not well-formatted");
    let documents: Vec<String> = dataset.documents;
    let queries: Vec<String> = dataset.queries;
    
    let document_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
    let queries_refs: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();

    let embedder = EmbedderBuilder::<u32>::with_fit_to_corpus(
        Language::English,
        &document_refs
    ).build();
    
    let embedded_documents = document_refs
        .iter()
        .map(|doc| embedder.embed(doc))
        .collect::<Vec<_>>();
    let embedded_queries = queries_refs
        .iter()
        .map(|query| embedder.embed(query))
        .collect::<Vec<_>>();
    
    let search_engine = SearchEngineBuilder::<u32>::with_corpus(Language::English, document_refs);
    
    
    for query in queries {
        let query_embedding = embedder.embed(&query);
    }
}