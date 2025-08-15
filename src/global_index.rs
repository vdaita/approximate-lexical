use::rayon::prelude::*;
use::std::collections::{BinaryHeap, HashMap};
use::std::cmp::Reverse;
use ordered_float::OrderedFloat;

use crate::sparse_to_dense::SparseToDense;
use crate::term_index::TermIndex;
use crate::utils::{ApproximateLexicalParameters, SegmentedCluster, ClusterSegment, IndexHeapEntry, ScoreHeapEntry, alpha_significance_compression};

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
    // 
    // requires a ClusterSegment because this will only be running over a single segment of the cluster per thread
    fn weighted_top_k_lists(query_weights: Vec<(usize, f32)>, cluster_segments: Vec<ClusterSegment>, top_k: usize) -> Vec<ScoreHeapEntry> {
        let weight_map: HashMap<usize, f32> = query_weights.iter().collect();
        
        
        let mut index_sorted_heap: BinaryHeap<IndexHeapEntry> = BinaryHeap::new();
        let mut score_sorted_heap: BinaryHeap<ScoreHeapEntry> = BinaryHeap::new();
        
        for (list_id, cluster_segments) in cluster_segments.iter().enumerate() {
            if cluster_segments.segment.len() > 0 {
                let (first_doc_id, first_term_score) = cluster_segments.segment[0];
                index_sorted_heap.push(IndexHeapEntry::new(
                    first_doc_id, 
                    first_term_score, 
                    list_id,
                    0 
                ));
            }
        }
        
        let mut current_index = usize::MAX;
        let mut current_score = 0.0;
        
        while let Some(index_heap_entry) = index_sorted_heap.pop() {
            if index_heap_entry.doc_id() == current_index {
                current_score += index_heap_entry.score() * token_weight_list_ordered[index_heap_entry.cluster_id()];
            } else {
                if current_index != usize::MAX {
                    score_sorted_heap.push(
                        ScoreHeapEntry::new(
                            current_index,
                            current_score
                        )
                    );
                    if score_sorted_heap.len() > top_k {
                        score_sorted_heap.pop();
                    }
                }
                current_index = index_heap_entry.doc_id();
                current_score = token_weight_list_ordered[index_heap_entry.cluster_id()] * index_heap_entry.score();
            }
            if index_heap_entry.segment_index() < cluster_segments[index_heap_entry.segment_index()].segment.len() - 1 {
                let (next_doc_id, next_term_score) = cluster_segments[index_heap_entry.segment_index()].segment[index_heap_entry.segment_index() + 1];
                index_sorted_heap.push(
                    IndexHeapEntry::new(
                        next_doc_id, 
                        next_term_score,
                        index_heap_entry.cluster_id(),
                        index_heap_entry.segment_index() + 1
                    )
                );
            }
        }
        
        let top_k_scores: Vec<ScoreHeapEntry> = score_sorted_heap.into_vec();
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