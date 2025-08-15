use rand::seq::SliceRandom;
use distances::Number;
use distances::vectors::{euclidean, euclidean_sq};
use rayon::prelude::*;

pub struct MiniBatchKMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub cluster_assignments: Vec<Vec<usize>>,
    pub num_clusters: usize,
    pub dim_size: usize
}

pub fn create_kmeans(
    data: &Vec<Vec<f32>>, 
    num_clusters: usize, 
    dim_size: usize, 
    kmeanspp_sample: usize, 
    kmeans_iterations: usize, 
    batch_size: usize,
    spherical: bool
) -> MiniBatchKMeansResult {
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(num_clusters);
    let mut rng = rand::thread_rng();

    if let Some(first) = data.choose(&mut rng) {
        centroids.push(first.clone());
        while centroids.len() < num_clusters {
            let sampled_points = data.choose_multiple(&mut rng, kmeanspp_sample).collect();
            let sampled_distances = sampled_points
                .par_iter()
                .map(|point| {
                    let mut min_dist = &centroids
                        .iter()
                        .map(|centroid| euclidean_sq(point, centroid))
                        .fold(f32::INFINITY, |a, b| a.min(b));                        
                    min_dist
                });
            let total_distance: f32 = sampled_distances.sum();
            let threshold = rand::random::<f32>() * total_distance;
            let mut cumulative: f32 = 0.0f32;
            for (point, dist) in sampled_points.iter().zip(sampled_distances) {
                cumulative += dist;
                if cumulative >= threshold {
                    centroids.push(point.clone());
                    break;
                }
            }
        }
    }

    // use minibatch kmeans to assign clusters and refine centroids
    for iter in 0..kmeans_iterations {
        let batch_dataset = data.choose_multiple(&mut rng, batch_size);
        let distances = batch_dataset
            .par_iter()
            .map(|point| {
                centroids
                    .iter()
                    .map(|centroid| euclidean(point, centroid))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();
        let labels = distances
            .par_iter()
            .map(|dists| dists.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap())
            .collect::<Vec<usize>>();
        let mut new_centroid = vec![0.0; dim_size];
        for centroid_index in 0..num_clusters {
            let cluster_points = batch_dataset
                .iter()
                .enumerate()
                .filter_map(|(i, point)| if labels[i] == centroid_index { Some(point) } else { None })
                .collect::<Vec<&Vec<f32>>>();
            if !cluster_points.is_empty() {
                new_centroid.iter_mut().for_each(|x| *x = 0.0);
                for dim_i in 0..dim_size {
                    for point in &cluster_points {
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
        .map(|point| {
            centroids
                .iter()
                .enumerate()
                .map(|(i, centroid)| (i, euclidean(point, centroid)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        })
        .collect::<Vec<usize>>();

    if spherical {
        centroids.iter_mut().for_each(|centroid| {
            let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
            centroid.iter_mut().for_each(|x| *x /= norm);
        });
    }

    MiniBatchKMeansResult {
        centroids: centroids,
        num_clusters: num_clusters,
        dim_size: dim_size,
        cluster_assignments: cluster_assignments
    }
}