use indicatif::ParallelProgressIterator;
use::rayon::prelude::*;
use::std::collections::{BinaryHeap, HashMap};
use::std::sync::Arc;
use::serde::{Serialize, Deserialize};
use log::{debug, trace};

use crate::sparse_to_dense::SparseToDense;
use crate::term_index::TermIndex;
use crate::utils::{
    alpha_significance_compression, ApproximateLexicalParameters, ClusterSegment, IndexHeapEntry,
    ScoreHeapEntry, SegmentedCluster,
};

#[derive(Serialize, Deserialize)]
pub struct GlobalIndex {
    term_indices: HashMap<usize, TermIndex>,
    projector: Arc<SparseToDense>,
    parameters: ApproximateLexicalParameters,
}

impl GlobalIndex {
    pub fn new(data: Vec<Vec<(usize, f32)>>, parameters: ApproximateLexicalParameters) -> Self {        
        let approximated_documents: Vec<(usize, Vec<(usize, f32)>)> = data
            .into_par_iter()
            .enumerate()
            .map(|(doc_id, doc)| {
                (doc_id, alpha_significance_compression(&doc, parameters.alpha_significance_threshold))
            })
            .collect();
        
        let unique_terms = approximated_documents
            .iter()
            .flat_map(|doc| doc.1.iter().map(|&(term, _)| term))
            .collect::<std::collections::HashSet<usize>>();
        let unique_terms_vec = unique_terms.iter().cloned().collect::<Vec<usize>>();
        println!("[GlobalIndex] finished creating SparseToDense projector with {} terms and {} dimensions.", (&unique_terms_vec).len(), parameters.dense_dim_size);

        let projector = Arc::new(SparseToDense::new(
            &unique_terms_vec,
            parameters.dense_dim_size
        ));

        let term_indices: HashMap<usize, TermIndex> = unique_terms_vec
            .into_par_iter()
            .progress()
            .map(|term_id| {
                let term_documents: Vec<&(usize, Vec<(usize, f32)>)> = approximated_documents
                    .iter()
                    .filter(|&(_, doc)| {
                        doc.iter().any(|&(t, _)| t == term_id)
                    })
                    .collect();
                
                // println!("[GlobalIndex] for term {}, there are {} documents", term_id, term_documents.len());
                
                let term_index = TermIndex::new(
                    term_id,
                    term_documents,
                    projector.clone(),
                    parameters.clone(),
                );
                
                (term_id, term_index)
            })
            .collect::<HashMap<usize, TermIndex>>();

        GlobalIndex {
            term_indices: term_indices,
            projector,
            parameters,
        }
    }

    // Takes a list of vectors and the token weights from the query, and produces top-k lists of document IDs and their scores
    // Intended for processing a single segment
    //
    // requires a ClusterSegment because this will only be running over a single segment of the cluster per thread
    pub fn weighted_top_k_lists(
        query_weights: &Vec<(usize, f32)>,
        cluster_segments: Vec<&ClusterSegment>,
        top_k: usize,
        segment_cluster_id: usize
    ) -> Vec<ScoreHeapEntry> {
        cluster_segments.iter().for_each(|cluster_segment| {
            println!("[weighted_top_k_lists={}] Processing ClusterSegment with id: {}, term_id: {}, length: {}", segment_cluster_id, cluster_segment.cluster_id, cluster_segment.term_id, cluster_segment.segment.len());
        });
        
        let token_weight_list_ordered: HashMap<usize, f32> = query_weights
            .iter()
            .map(|&(token_id, weight)| (token_id, weight))
            .collect();

        let mut index_sorted_heap: BinaryHeap<IndexHeapEntry> = BinaryHeap::new();
        let mut score_sorted_heap: BinaryHeap<ScoreHeapEntry> = BinaryHeap::new();

        for (list_id, cluster_segments) in cluster_segments.iter().enumerate() {
            if cluster_segments.segment.len() > 0 {
                let (first_doc_id, first_term_score) = cluster_segments.segment[0];
                index_sorted_heap.push(IndexHeapEntry::new(
                    first_doc_id,
                    first_term_score,
                    cluster_segments.term_id,
                    list_id,
                    0,
                ));
            }
        }

        let mut current_index = usize::MAX;
        let mut current_score = 0.0;

        while let Some(index_heap_entry) = index_sorted_heap.pop() {
            let cluster_id: usize = index_heap_entry.cluster_id();
            let doc_id: usize = index_heap_entry.doc_id();
            let token_id: usize = index_heap_entry.token_id();
            let score: f32 = index_heap_entry.score();
            
            println!(
                "[weighted_top_k_lists={} -> while loop] Processing doc_id: {}, score: {}, cluster_id: {}, token_id: {}, segment_index: {}",
                segment_cluster_id, doc_id, score, cluster_id, token_id, index_heap_entry.segment_index()
            );
            
            let Some(token_weight) = token_weight_list_ordered.get(&token_id) else {
                println!(
                    "[weighted_top_k_lists={} -> while loop] Processing cluster_id: {}, token {} not in query",
                    segment_cluster_id,
                    cluster_id,
                    token_id
                );
                continue; // skip if the cluster is not in the query
            };

            if doc_id == current_index {
                current_score += token_weight * score;
            } else {
                if current_index != usize::MAX {
                    score_sorted_heap.push(ScoreHeapEntry::new(current_index, current_score));
                    if score_sorted_heap.len() > top_k {
                        score_sorted_heap.pop();
                    }
                }
                current_index = doc_id;
                current_score = token_weight * score;
            }
            if index_heap_entry.segment_index() + 1
                < cluster_segments[index_heap_entry.segment_index()]
                    .segment
                    .len()
            {
                let cluster_segment = &cluster_segments[index_heap_entry.segment_index()];
    
                let (next_doc_id, next_term_score) = cluster_segment.segment[index_heap_entry.segment_index() + 1];
                let term_id = cluster_segment.term_id;
                
                index_sorted_heap.push(IndexHeapEntry::new(
                    next_doc_id,
                    next_term_score,
                    term_id,
                    index_heap_entry.cluster_id(),
                    index_heap_entry.segment_index() + 1,
                ));
            }
        }

        let top_k_scores: Vec<ScoreHeapEntry> = score_sorted_heap.into_vec();
        top_k_scores
    }

    // read the top-k clusters, process them across different segments with different threads, and return the top-k elements
    pub fn query(
        &self,
        query: &Vec<(usize, f32)>,
        top_k_clusters: usize,
        top_k_elements: usize,
    ) -> Vec<(usize, f32)> {
        let projected_vector = self.projector.project(query);
        
        let query_weight_sum = query.iter().map(|&(_, value)| value).sum::<f32>();
        let normalized_query = query
            .iter()
            .map(|&(term, value)| (term, value / query_weight_sum))
            .collect::<Vec<(usize, f32)>>();
        let top_k_segmented_clusters: Vec<&SegmentedCluster> = normalized_query
            .par_iter()
            .map(|&(term_id, weight)| {
                let num_clusters = ((weight * (top_k_clusters as f32)).floor()) as usize;
                let Some(term_index) = self.term_indices.get(&term_id) else {
                    debug!("Term ID {} not found in term indices", term_id);
                    return vec![];
                };
                term_index
                    .get_segmented_clusters(&projected_vector, num_clusters)
            })
            .flatten_iter()
            .collect();
        
        debug!(
            "[GlobalIndex] Found {} segmented clusters for the query",
            top_k_segmented_clusters.len()
        );
        
        let chunk_scores: Vec<(usize, f32)> = (0..self.parameters.num_cluster_segments)
            .into_par_iter()
            .map(|cluster_segment| {
                let cluster_segments: Vec<&ClusterSegment> = top_k_segmented_clusters
                    .iter()
                    .map(|segmented_cluster| &segmented_cluster.segments[cluster_segment])
                    .collect();
                
                trace!(
                    "[GlobalIndex] Processing segment {} with {} clusters",
                    cluster_segment,
                    cluster_segments.len()
                );

                let top_k_score_entries: Vec<ScoreHeapEntry> =
                    GlobalIndex::weighted_top_k_lists(query, cluster_segments, top_k_elements, cluster_segment);
                
                debug!(
                    "[GlobalIndex] Found {} top-k score entries in segment {}",
                    top_k_score_entries.len(),
                    cluster_segment
                );

                let scores: Vec<(usize, f32)> = top_k_score_entries
                    .into_iter()
                    .map(|entry| (entry.doc_id(), entry.score()))
                    .collect();

                scores
            })
            .flatten()
            .collect();

        chunk_scores
    }
}
