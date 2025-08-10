use std::collections::{HashMap};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rand::distr::weighted::WeightedIndex;
use rand::thread_rng;
use rand::distr::Distribution;

#[derive(Debug, Clone)]
pub struct SampleSearcher {
    pub prob_map: HashMap<usize, (Vec<usize>, WeightedIndex<f32>)>,
    micro: bool
}

impl SampleSearcher {
    pub fn new(documents: Vec<Vec<(usize, f32)>>, micro: bool) -> Self {
        println!("[SampleSearcher] SampleSearcher::new called with {} documents, micro = {}", documents.len(), micro);

        let pb = ProgressBar::new(documents.len() as u64);
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .progress_chars("#>-");
        pb.set_style(style);

        let mut weight_map: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (doc_idx, doc) in documents.iter().enumerate() {
            for &(term_idx, weight) in doc {
                weight_map.entry(term_idx).or_default().push((doc_idx, weight));                
            }
            pb.inc(1);
        }
        pb.finish_with_message("Index construction complete");

        let prob_map: HashMap<usize, (Vec<usize>, WeightedIndex<f32>)> = weight_map.into_iter()
            .map(|(term_idx, weights)| {
                let weighted_index = WeightedIndex::new(weights.iter().map(|&(_, weight)| weight).collect::<Vec<f32>>()).unwrap(); 
                let doc_ids = weights.iter().map(|&(doc_idx, _)| doc_idx).collect();
                (term_idx, (doc_ids, weighted_index))
            })
            .collect();
        
        Self {
            prob_map,
            micro
        }
    }

    pub fn search(&self, query: &[(usize, u32)], top_k: usize, B: usize, num_threads: usize) -> Vec<usize> { 
        if self.micro { 
            println!("[SampleSearcher] SampleSearcher::search called with top_k = {}", top_k);
        }
        let query_weighted_index = WeightedIndex::new(query.iter().map(|&(_, weight)| weight as f32)).unwrap();

        let local_hashmaps: Vec<HashMap<usize, usize>> = (0..num_threads).into_par_iter().map(|_| {
            let mut local_results = HashMap::new();
            let mut rng = thread_rng();

            for _ in (0..B).step_by(num_threads) {
                let term_idx = query_weighted_index.sample(&mut rng);
                if let Some((doc_ids, weighted_index)) = self.prob_map.get(&term_idx) {
                    let doc_sample_idx = weighted_index.sample(&mut rng);
                    let doc_id = doc_ids[doc_sample_idx];
                    *local_results.entry(doc_id).or_insert(0 as usize) += 1;
                }
            }

            local_results
        }).collect();

        let mut global_results = HashMap::new();
        for local in local_hashmaps {
            for (doc_id, count) in local {
                *global_results.entry(doc_id).or_insert(0) += count;
            }
        }

        // Get the top_k doc_ids with the most occurrences
        let mut doc_counts: Vec<(usize, usize)> = global_results.into_iter().collect();
        doc_counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        doc_counts.truncate(top_k);

        doc_counts.into_iter().map(|(doc_id, _)| doc_id).collect()
    }
}