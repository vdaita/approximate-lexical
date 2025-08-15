use crate::sparse_to_dense::SparseToDense;
use std::collections::HashMap;
use crate::utils::{ApproximateLexicalParameters, ClusterSegment, SegmentedCluster};
use crate::kmeans::{MiniBatchKMeansResult, create_kmeans};
use std::sync::Arc;

pub struct TermIndex<'a> {
    dense_centroids: Vec<Vec<f32>>,
    segmented_clusters: Vec<SegmentedCluster>,
    projector: Arc<SparseToDense>, // keep the same one for every sub-index
}

impl TermIndex {
    pub fn new(term: usize, documents: &Vec<Vec<(usize, f32)>>, projector: Arc<SparseToDense>, parameters: ApproximateLexicalParameters) -> Self {
        let projected_documents = projector.project_multiple(documents);
        let kmeans_result = create_kmeans(
            &projected_documents, 
            parameters.num_clusters, 
            parameters.dense_dim_size, 
            parameters.kmeanspp_sample, 
            parameters.kmeans_iterations, 
            parameters.batch_size,
            parameters.spherical
        );
        
        let centroids = kmeans_result.centroids;
        let mut segmented_clusters: Vec<SegmentedCluster> = kmeans_result.cluster_assignments
            .into_iter()
            .enumerate()
            .map(|(cluster_id, segment_ids)| {
                let ids: Vec<usize> = segment_ids.iter().map(|&id| id).collect();
                let values_map: HashMap<usize, f32> = ids.iter().map(|&id| (id, 1.0)).collect(); // Assuming all values are 1.0 for simplicity
                TermIndex::format_cluster(term, cluster_id, &ids, &values_map, parameters, documents.len())
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
        let segment_size = ids.len() / num_segments;

        for i in 0..num_segments {
            let start = i * segment_size;
            let end = if i == num_segments - 1 { ids.len() } else { start + segment_size };
            let segment_ids: Vec<usize> = ids.iter()
                .filter(|&&id| {
                    if let Some(&value) = values_map.get(&id) {
                        value >= start as f32 && value < end as f32
                    } else {
                        false
                    }
                })
                .cloned()
                .collect();
            
            segments.push(ClusterSegment {
                cluster_id: i,
                term_id: 0, // This should be set to the actual term ID
                segment_id: i,
                segment: segment_ids.to_vec(),
            });
        }

        SegmentedCluster {
            cluster_id: cluster_id,
            term_id: term_id, 
            num_segments: num_segments,
            segments: segments
        }
    }
  
    pub fn get_segmented_clusters(&self, projected_query: &Vec<f32>) -> Vec<&SegmentedCluster>{
        // Find the closest cluster and return a list of relevant SegmentedClusters
    }
}