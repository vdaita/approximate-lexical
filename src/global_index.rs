use::rayon::prelude::*;
use::std::collections::{BinaryHeap, HashMap};
use::std::sync::Arc;
use::serde::{Serialize, Deserialize};

use crate::sparse_to_dense::SparseToDense;
use crate::term_index::TermIndex;
use crate::utils::{
    alpha_significance_compression, ApproximateLexicalParameters, ClusterSegment, IndexHeapEntry,
    ScoreHeapEntry, SegmentedCluster,
};

#[derive(Serialize, Deserialize)]
pub struct GlobalIndex {
    term_indices: Vec<TermIndex>,
    projector: Arc<SparseToDense>,
    parameters: ApproximateLexicalParameters,
}

impl GlobalIndex {
    pub fn new(data: Vec<Vec<(usize, f32)>>, parameters: ApproximateLexicalParameters) -> Self {
        let max_term_index = data
            .par_iter()
            .flat_map_iter(|doc| doc.iter().map(|&(term, _)| term))
            .max()
            .unwrap_or(0);
        
        let projector = Arc::new(SparseToDense::new(
            max_term_index,
            parameters.dense_dim_size
        ));
        

        if max_term_index == 0 {
            println!("[GlobalIndex] no terms found in the data.");
        }

        let approximated_documents: Vec<Vec<(usize, f32)>> = data
            .into_par_iter()
            .map(|doc| {
                alpha_significance_compression(&doc, parameters.alpha_significance_threshold)
            })
            .collect();

        let term_indices: Vec<TermIndex> = (0..max_term_index)
            .into_par_iter()
            .map(|term_index| {
                let term_documents: Vec<Vec<(usize, f32)>> = approximated_documents
                    .par_iter()
                    .filter_map(|doc| {
                        doc.iter()
                            .find(|&&(t, _)| t == term_index)
                            .map(|&(t, v)| vec![(t, v)]) // Collect only the term and its value
                    })
                    .collect();
                TermIndex::new(
                    term_index,
                    &term_documents,
                    projector.clone(),
                    parameters.clone(),
                )
            })
            .collect();

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
    ) -> Vec<ScoreHeapEntry> {
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
            let score: f32 = index_heap_entry.score();
            let Some(token_weight) = token_weight_list_ordered.get(&cluster_id) else {
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
            if index_heap_entry.segment_index()
                < cluster_segments[index_heap_entry.segment_index()]
                    .segment
                    .len()
                    - 1
            {
                let (next_doc_id, next_term_score) = cluster_segments
                    [index_heap_entry.segment_index()]
                .segment[index_heap_entry.segment_index() + 1];
                index_sorted_heap.push(IndexHeapEntry::new(
                    next_doc_id,
                    next_term_score,
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
            .map(|&(term_index, weight)| {
                let num_clusters = ((weight * (top_k_clusters as f32)).floor()) as usize;
                self.term_indices[term_index]
                    .get_segmented_clusters(&projected_vector, num_clusters)
            })
            .flatten_iter()
            .collect();

        let chunk_scores: Vec<(usize, f32)> = (0..self.parameters.num_cluster_segments)
            .into_par_iter()
            .map(|cluster_segment| {
                let cluster_segments: Vec<&ClusterSegment> = top_k_segmented_clusters
                    .iter()
                    .map(|segmented_cluster| &segmented_cluster.segments[cluster_segment])
                    .collect();

                let top_k_score_entries: Vec<ScoreHeapEntry> =
                    GlobalIndex::weighted_top_k_lists(query, cluster_segments, top_k_elements);

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
