use distances::vectors::{euclidean, euclidean_sq};
use rand::prelude::IndexedRandom;
use rayon::prelude::*;

use crate::{ApproximateLexicalParameters};

pub struct MiniBatchKMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub cluster_assignments: Vec<Vec<usize>>,
    pub num_clusters: usize,
    pub dim_size: usize,
}

pub fn create_kmeans(
    data: &Vec<(usize, Vec<f32>)>,
    parameters: &ApproximateLexicalParameters,
) -> MiniBatchKMeansResult {
    let num_clusters: usize = (data.len() / parameters.cluster_size) + 1;
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(num_clusters);
    let mut rng = rand::rng();
    
    if let Some((_first_index, first_point)) = data.choose(&mut rng) {
        centroids.push(first_point.clone());
        while centroids.len() < num_clusters {
            let sampled_points: Vec<&(usize, Vec<f32>)> = data
                .choose_multiple(&mut rng, parameters.kmeanspp_sample)
                .into_iter()
                .collect();
            let sampled_distances: Vec<f32> = sampled_points
                .par_iter()
                .map(|&(_, point)| {
                    let min_dist: f32 = centroids
                        .iter()
                        .map(|centroid| euclidean_sq(point, centroid))
                        .fold(f32::INFINITY, |a, b| a.min(b));

                    min_dist
                })
                .collect();

            let total_distance: f32 = sampled_distances.iter().sum();
            let threshold = rand::random::<f32>() * total_distance;
            let mut cumulative: f32 = 0.0f32;

            for (&(_, point), &dist) in sampled_points.iter().zip(sampled_distances.iter()) {
                cumulative += dist; // dist is now f32, not &f32
                if cumulative >= threshold {
                    centroids.push(point.clone().to_vec());
                    break;
                }
            }
        }
    }

    // use minibatch kmeans to assign clusters and refine centroids
    for _iter in 0..parameters.kmeans_iterations {
        let batch_dataset: Vec<&(usize, Vec<f32>)> = data
            .choose_multiple(&mut rng, parameters.kmeans_batch_size)
            .collect();
        let distances = batch_dataset
            .par_iter()
            .map(|&(_, point)| {
                centroids
                    .iter()
                    .map(|centroid| euclidean(point, centroid))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();
        let labels: Vec<usize> = distances
            .par_iter()
            .map(|dists| {
                dists
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap()
            })
            .collect::<Vec<usize>>();
        for centroid_index in 0..num_clusters {
            let cluster_points = batch_dataset
                .iter()
                .zip(labels.iter())
                .filter(|(_, &label)| label == centroid_index)
                .map(|(point, _)| *point) // iter works by passing a reference, so before collecting, you need to dereference so that you get the right type
                .collect::<Vec<&(usize, Vec<f32>)>>();
            let mut new_centroid = vec![0.0; parameters.dense_dim_size];
            if !cluster_points.is_empty() {
                new_centroid.iter_mut().for_each(|x| *x = 0.0);
                for dim_i in 0..parameters.dense_dim_size {
                    if dim_i >= parameters.dense_dim_size {
                        panic!(
                            "Dimension index out of bounds: {} for dimension size {}",
                            dim_i, parameters.dense_dim_size
                        );
                    }
                    for (doc_id, point) in &cluster_points {
                        if point.len() != parameters.dense_dim_size {
                            panic!(
                                "Point dimension mismatch: expected {}, got {}",
                                parameters.dense_dim_size,
                                point.len()
                            );
                        }
                        new_centroid[dim_i] += point[dim_i];
                    }
                    new_centroid[dim_i] /= cluster_points.len() as f32;
                }
                centroids[centroid_index] = new_centroid.clone();
            }
        }
    }

    let cluster_assignments = data
        .par_iter()
        .map(|(doc_id, point)| {
            centroids
                .iter()
                .enumerate()
                .map(|(i, centroid)| (i, euclidean::<f32, f32>(point, centroid)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| (doc_id.clone(), i))
                .unwrap()
        })
        .collect::<Vec<(usize, usize)>>();

    let centroid_to_doc_ids = centroids
        .iter()
        .enumerate()
        .map(|(i, _)| {
            cluster_assignments
                .iter()
                .filter_map(|(doc_id, cluster_id)| {
                    if *cluster_id == i {
                        Some(*doc_id)
                    } else {
                        None
                    }
                })
                .collect::<Vec<usize>>()
        })
        .collect(); 

    if parameters.spherical {
        centroids.iter_mut().for_each(|centroid| {
            let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
            centroid.iter_mut().for_each(|x| *x /= norm);
        });
    }

    MiniBatchKMeansResult {
        centroids: centroids,
        num_clusters: num_clusters,
        dim_size: parameters.dense_dim_size,
        cluster_assignments: centroid_to_doc_ids,
    }
}
