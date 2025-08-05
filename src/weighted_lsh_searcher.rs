use std::collections::{HashMap};
use rand_distr::{Distribution, Gamma, Uniform};
use rand::rngs::StdRng;
use rand::{SeedableRng};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

pub fn hash(i: usize, j: u32, seed: u32) -> u64 {
    (((i as u32) * 71 + j * 23 + seed * 7)) as u64
}

pub fn gamma_sample(i: usize, j: u32, seed: u32, gamma: &Gamma<f32>) -> f32 {
    let mut rng = StdRng::seed_from_u64(hash(i, j, seed));
    gamma.sample(&mut rng)
}

pub fn uniform_sample(i: usize, j: u32, seed: u32, uniform: &Uniform<f32>) -> f32 {
    let mut rng = StdRng::seed_from_u64(hash(i, j, seed));
    uniform.sample(&mut rng)
}

fn hash_band(vector: &[(usize, f32)], band: usize, r_seed: u32, c_seed: u32, b_seed: u32, gamma: &Gamma<f32>, uniform: &Uniform<f32>, micro: bool) -> usize {
    let t: Vec<(usize, f32)> = vector.iter().map(|&(index, weight)| {
        let r_ij = gamma_sample(index, band as u32, r_seed, gamma);
        let b_ij = uniform_sample(index, band as u32, b_seed, uniform);
        let t_ij = ((weight.ln() / r_ij) + b_ij).floor();
        (index, t_ij)
    }).collect();

    let y: Vec<(usize, f32)> = t.iter().map(|&(index, t_ij)| {
        let b_ij = uniform_sample(index, band as u32, b_seed, uniform);
        let r_ij = gamma_sample(index, band as u32, r_seed, gamma);

        let y_ij = (r_ij * (t_ij - b_ij)).exp();
        (index, y_ij)
    }).collect();

    let a: Vec<(usize, f32)> = y.iter().map(|&(index, y_ij)| {
        let c_ij = gamma_sample(index, band as u32, c_seed, gamma);
        let r_ij = gamma_sample(index, band as u32, r_seed, gamma);

        let a_ij = c_ij / (y_ij * r_ij.exp());
        (index, a_ij)
    }).collect();

    let min_index = a.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|&(index, _)| index);
    min_index.expect("At least one element should be present in the vector")
}


#[derive(Debug, Clone)]
pub struct WeightedMinHashLSH {
    pub hashmaps: Vec<HashMap<usize, Vec<usize>>>, // create a map from band to doc ids
    num_bands: usize,

    gamma: Gamma<f32>,
    uniform: Uniform<f32>,
    r_seeds: Vec<u32>,
    c_seeds: Vec<u32>,
    b_seeds: Vec<u32>,

    micro: bool,
}

impl WeightedMinHashLSH {
    pub fn new(scored_documents: &Vec<Vec<(usize, f32)>>, num_bands: usize, seed: u32, micro: bool) -> Self {
        if micro {
            println!("[micro] WeightedMinHashLSH::new called");
        }

        let gamma = Gamma::new(2.0, 1.0).unwrap();
        let uniform = Uniform::new(0.0, 1.0).unwrap();

        let r_seeds: Vec<u32> = (0..num_bands).map(|i| seed + i as u32).collect();
        let c_seeds: Vec<u32> = (0..num_bands).map(|i| seed + i as u32 + num_bands as u32).collect();
        let b_seeds: Vec<u32> = (0..num_bands).map(|i| seed + i as u32 + num_bands as u32 * 2).collect();

        // Progress bar setup
        let pb = ProgressBar::new((scored_documents.len() * num_bands) as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .progress_chars("#>-")
        );
        
        let hashmaps: Vec<HashMap<usize, Vec<usize>>> = (0..num_bands)
            .into_par_iter()
            .map(|band| {
                let mut band_map: HashMap<usize, Vec<usize>> = HashMap::new();
                for (doc_id, vector) in scored_documents.iter().enumerate() {
                    let hash = hash_band(&vector, band, r_seeds[band], c_seeds[band], b_seeds[band], &gamma, &uniform, micro);
                    band_map.entry(hash).or_default().push(doc_id);
                    pb.inc(1);
                }
                band_map
            })
            .collect();

        pb.finish_with_message("Documents loaded.");

        Self {
            hashmaps,
            num_bands,
            gamma,
            uniform,
            r_seeds,
            c_seeds,
            b_seeds,
            micro,
        }
    }

    pub fn query(self: &Self, vector: Vec<(usize, f32)>) -> Vec<usize> {
        if self.micro {
            println!("[micro] WeightedMinHashLSH::query called");
        }
        let mut results = Vec::new();
        for band in 0..self.num_bands {
            let hash_value = hash_band(&vector, band, self.r_seeds[band], self.c_seeds[band], self.b_seeds[band], &self.gamma, &self.uniform, self.micro);
            if let Some(doc_ids) = self.hashmaps[band].get(&hash_value) {
                results.extend(doc_ids.iter().cloned());
                if self.micro {
                    println!("[micro] Found {} documents for band {} with hash {}", doc_ids.len(), band, hash_value);
                }
            } else {
                if self.micro {
                    println!("[micro] No documents found for band {} with hash {}", band, hash_value);
                }
            }
        }
        results.sort_unstable();
        results.dedup();
        results
    }
}