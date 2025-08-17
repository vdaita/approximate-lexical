use ordered_float::OrderedFloat;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use std::cmp::Reverse;

// alpha-significance to reduce size of posting lists
pub fn alpha_significance_compression(
    data: &Vec<(usize, f32)>,
    alpha_threshold: f32,
) -> Vec<(usize, f32)> {
    let mut compressed = Vec::new();
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let alpha_sum = sorted_data.iter().map(|&(_, score)| score).sum::<f32>();
    let mut current_sum = 0.0;
    for &(doc_id, score) in &sorted_data {
        current_sum += score;
        if current_sum / alpha_sum >= alpha_threshold {
            compressed.push((doc_id, score));
        }
    }
    compressed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproximateLexicalParameters {
    pub num_clusters: usize,
    pub dense_dim_size: usize,
    pub kmeanspp_sample: usize,
    pub kmeans_iterations: usize,
    pub kmeans_batch_size: usize,
    pub spherical: bool,
    pub num_cluster_segments: usize,
    pub alpha_significance_threshold: f32,
}

impl ApproximateLexicalParameters {
    pub fn new(
        num_clusters: usize,
        dense_dim_size: usize,
        kmeanspp_sample: usize,
        kmeans_iterations: usize,
        kmeans_batch_size: usize,
        spherical: bool,
        num_cluster_segments: usize,
        alpha_significance_threshold: f32,
    ) -> Self {
        ApproximateLexicalParameters {
            num_clusters,
            dense_dim_size,
            kmeanspp_sample,
            kmeans_iterations,
            kmeans_batch_size,
            spherical,
            num_cluster_segments,
            alpha_significance_threshold,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SegmentedCluster {
    pub cluster_id: usize,
    pub term_id: usize,
    pub num_segments: usize,
    pub segments: Arc<Vec<ClusterSegment>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClusterSegment {
    pub cluster_id: usize,
    pub term_id: usize,
    pub segment_id: usize,
    pub segment: Arc<Vec<(usize, f32)>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct IndexHeapEntry {
    pub doc_id: Reverse<usize>,
    pub score: OrderedFloat<f32>,
    pub cluster_id: usize,
    pub segment_index: usize,
}

impl IndexHeapEntry {
    pub fn new(doc_id: usize, score: f32, cluster_id: usize, segment_index: usize) -> Self {
        IndexHeapEntry {
            doc_id: Reverse(doc_id),
            score: OrderedFloat(score),
            cluster_id,
            segment_index, // describes where in the segment you currently are
        }
    }

    pub fn score(&self) -> f32 {
        self.score.into_inner()
    }

    pub fn doc_id(&self) -> usize {
        self.doc_id.0
    }

    pub fn cluster_id(&self) -> usize {
        self.cluster_id
    }

    pub fn segment_index(&self) -> usize {
        self.segment_index
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct ScoreHeapEntry {
    pub score: OrderedFloat<f32>,
    pub doc_id: usize,
}

impl ScoreHeapEntry {
    pub fn new(doc_id: usize, score: f32) -> Self {
        ScoreHeapEntry {
            doc_id,
            score: OrderedFloat(score),
        }
    }

    pub fn score(&self) -> f32 {
        self.score.into_inner()
    }

    pub fn doc_id(&self) -> usize {
        self.doc_id
    }
}
