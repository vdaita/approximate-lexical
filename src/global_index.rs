use::rayon::prelude::*;
use::std::collections::BinaryHeap;
use::std::cmp::Reverse;
use ordered_float::OrderedFloat;

use crate::sparse_to_dense::SparseToDense;
use crate::term_index::TermIndex;
use crate::utils::{ApproximateLexicalParameters, SegmentedCluster, alpha_significance_compression};

struct GlobalIndex {
    term_indices: Vec<TermIndex>,
    parameters: ApproximateLexicalParameters
}

impl GlobalIndex {
    pub fn new() -> Self {
        GlobalIndex {
            term_indices: Vec::new(),
        }
    }
    
    // Takes a list of vectors and the token weights from the query, and produces top-k lists of document IDs and their scores
    // Intended for processing a single segment
    fn weighted_top_k_lists(query_weights: Vec<(usize, f32)>, segmented_clusters: Vec<SegmentedCluster>, top_k: usize) -> Vec<(OrderedFloat<f32>, usize)> {
        let token_weight_list_ordered: Vec<f32> = segmented_clusters
            .iter()
            .map(|&segmented_cluster| {
                let token_weight = query_weights
                    .iter()
                    .find(|&&(id, _)| id == segmented_cluster.term_id)
                    .map_or(0.0, |&(_, weight)| weight);
                token_weight
            })
            .collect();
        
        let mut index_sorted_heap: BinaryHeap<(Reverse<usize>, OrderedFloat<f32>, usize, usize)> = BinaryHeap::new();
        let mut score_sorted_heap: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        
        for (list_id, segmented_cluster) in segmented_clusters.iter().enumerate() {
            if segmented_cluster.segments.len() > 0 {
                let first_doc_id = segmented_cluster.segments[]
                
                let (first_doc_id, first_term_score) = segmented_cluster[0];
                index_sorted_heap.push((Reverse(first_doc_id), OrderedFloat(first_term_score), 0, list_id));
            }
        }
        
        let mut current_index = usize::MAX;
        let mut current_score = 0.0;
        
        while let Some((Reverse(doc_id), OrderedFloat(term_score), list_index, list_id)) = index_sorted_heap.pop() {
            if doc_id == current_index {
                current_score += token_weight_list_ordered[list_id] * term_score;
            } else {
                if current_index != usize::MAX {
                    score_sorted_heap.push((OrderedFloat(current_score), current_index));
                    if score_sorted_heap.len() > top_k {
                        score_sorted_heap.pop();
                    }
                }
                current_index = doc_id;
                current_score = token_weight_list_ordered[list_id] * term_score;
            }
            if list_id < lists[list_id].1.len() - 1 {
                let (next_doc_id, next_term_score) = lists[list_id].1[list_index + 1];
                index_sorted_heap.push((Reverse(next_doc_id), OrderedFloat(next_term_score), list_index + 1, list_id));
            }
        }
        
        let top_k_scores: Vec<(OrderedFloat<f32>, usize)> = score_sorted_heap.into_vec();
        top_k_scores
    }
    
    // read the top-k clusters, process them across different segments with different threads, and return the top-k elements
    fn query(&self, query: &[(usize, f32)], top_k_clusters: usize, top_k_elements: usize) -> Vec<usize> {
        let projected_vector = self.sparse_to_dense.project(query);
        
        let top_k_cluster_pointers: Vec<(usize, Vec<&Vec<(usize, f32)>>)> = self.term_indices
            .par_iter()
            .map(|term_index| (term_index, term_index.query(&projected_vector)))
            .collect();
        
        let chunk_scores: Vec<Vec<(OrderedSize<f32>, usize)>> = 0..self.parameters.num_cluster_segments
            .par_iter()
            .map(|cluster_segment| {
                let mut lists: Vec<(usize, &Vec<(usize, f32)>)> = Vec::new();
                
                
                for (term_index, term_scores) in top_k_cluster_pointers.iter() {
                    lists.push((term_index, term_scores[cluster_segment]));
                }
                GlobalIndex::weighted_top_k_lists(query, lists, top_k_elements)
            })
    }
    
}