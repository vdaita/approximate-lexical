use crate::sparse_to_dense::SparseToDense;

pub struct TermIndex {
    dense_clusters: Vec<Vec<f32>>,
    cluster_scores: Vec<Vec<Vec<(usize, f32)>>>,
    projector: &SparseToDense, // keep the same one for every sub-index
}

impl TermIndex {
    pub fn new() -> Self {
        TermIndex {
            dense_cluster: Vec::new(),
            cluster_indices: Vec::new()
        }
    }
    
    // create subclusters and use kmeans within the term 
    // subclusters get grouped into cluster segments

    // sort the scores in everything by ascending order
    fn sort_cluster_scores(&mut self){
        for cluster in &mut self.cluster_scores {
            for subchunk in cluster {
                subchunk.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
    }

    // TODO: write a function that will do the clustering
    pub fn query(&self, vector: &Vec<f32>) -> Vec<&Vec<(usize, f32)>> {
        let projected = self.projector(vector);
        let cluster_ids = self.get_top_k_clusters(&projected);
    }
}