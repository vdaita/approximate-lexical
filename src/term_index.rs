use crate::sparse_to_dense::SparseToDense;
use std::collections::HashMap;
use crate::utils::{ApproximateLexicalParameters, ClusterSegment, SegmentedCluster};
use crate::kmeans::{create_kmeans};
use std::sync::Arc;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TermIndex {
    dense_centroids: Vec<Vec<f32>>,
    segmented_clusters: Vec<SegmentedCluster>,
    projector: Arc<SparseToDense>, // keep the same one for every sub-index
}

impl TermIndex {
    pub fn new(term: usize, documents: &Vec<Vec<(usize, f32)>>, projector: Arc<SparseToDense>, parameters: ApproximateLexicalParameters) -> Self {
        let projected_documents = projector.project_multiple(documents);
        let kmeans_result = create_kmeans(
            &projected_documents, 
            &parameters
        );
        
        let centroids = kmeans_result.centroids;
        let segmented_clusters: Vec<SegmentedCluster> = kmeans_result.cluster_assignments
            .into_iter()
            .enumerate()
            .map(|(cluster_id, segment_ids)| {
                let ids: Vec<usize> = segment_ids.iter().map(|&id| id).collect();
                let values_map: HashMap<usize, f32> = ids.iter().map(|&id| (id, 1.0)).collect(); // Assuming all values are 1.0 for simplicity
                TermIndex::format_cluster(term, cluster_id, &ids, &values_map, parameters.clone(), documents.len())
            })
            .collect();
        
        TermIndex {
            dense_centroids: centroids,
            segmented_clusters: segmented_clusters,
            projector: projector,
        }
    }
    
    // create subclusters and use kmeans within the term 
    // subclusters get grouped into cluster segments
    fn format_cluster(term_id: usize, cluster_id: usize, ids: &Vec<usize>, values_map: &HashMap<usize, f32>, parameters: ApproximateLexicalParameters, max_document_id: usize) -> SegmentedCluster {
        let mut segments: Vec<ClusterSegment> = Vec::new();
        let num_segments = parameters.num_cluster_segments;
        let segment_size = max_document_id / num_segments;

        for i in 0..num_segments {
            let start = i * segment_size;
            let end = if i == num_segments - 1 { max_document_id } else { start + segment_size };

            // Simply slice the ids vector based on position instead of using values_map filtering
            let segment_ids: Vec<usize> = ids[start..end].to_vec();

            let segment_id_value_pairs: Vec<(usize, f32)> = segment_ids.iter()
                .map(|&id| (id, values_map.get(&id).copied().unwrap_or(1.0)))
                .collect();

            segments.push(ClusterSegment {
                cluster_id: i,
                term_id: term_id,
                segment_id: i,
                segment: Arc::new(segment_id_value_pairs)
            });
        }

        SegmentedCluster {
            cluster_id: cluster_id,
            term_id: term_id, 
            num_segments: num_segments,
            segments: Arc::new(segments)
        }
    }
  
    pub fn get_segmented_clusters(&self, projected_query: &Vec<f32>, top_k: usize) -> Vec<&SegmentedCluster>{
        // Find the closest cluster and return the most relevant segmented clusters
        let mut cluster_scores = self.dense_centroids.par_iter()
            .enumerate()
            .map(|(i, centroid)| {
                let score = centroid.iter().zip(projected_query).map(|(c, q)| c * q).sum::<f32>();
                (i, score)
            })
            .collect::<Vec<_>>();

        // Select the top-k clusters based on scores
        cluster_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_cluster_ids: Vec<usize> = cluster_scores.iter()
            .take(top_k)
            .map(|(id, _)| *id)
            .collect();

        // Select the segmented cluster corresponding to each of the cluster ids
        top_cluster_ids.iter()
            .filter_map(|&cluster_id| self.segmented_clusters.get(cluster_id))
            .collect()
    }
}