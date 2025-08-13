use std::cmp::Reverse;
use ordered_float::OrderedFloat;

// alpha-significance to reduce size of posting lists
pub fn alpha_significance_compression(data: &[(u32, f32)], alpha_threshold: f32) -> Vec<(u32, f32)> {
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

pub struct ApproximateLexicalParameters {
    pub num_clusters: usize, 
    pub dense_dim_size: usize,
    pub kmeanspp_sample: usize,
    pub kmeans_iterations: usize,
    pub batch_size: usize,
    pub spherical: bool,
    pub num_cluster_segments: usize
}

pub struct SegmentedCluster<'a> {
    pub cluster_id: usize, 
    pub term_id: usize,
    pub num_segments: usize,
    pub segments: Vec<ClusterSegment<'a>>
}

pub struct ClusterSegment<'a> {
    pub cluster_id: usize,
    pub term_id: usize,
    pub segment_id: usize,
    pub segment: &'a Vec<(usize, f32)>
}

#[derive(Debug, Clone, Ord, ParialOrd)]
pub struct IndexHeapEntry {
    pub index: usize,
    pub score: f32,
    pub cluster_id: usize,
    pub segment_id: usize
}

impl Ord for IndexHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(std::cmp::Ordering::Equal)
    }
}