use std::collections::{HashMap};
use rand_distr::{Distribution, Gamma, Uniform};
use rand::rngs::StdRng;
use rand::{SeedableRng, Rng};

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
        let mut hashmaps: Vec<HashMap<usize, Vec<usize>>> = Vec::with_capacity(num_bands);
        for i in 0..num_bands {
            if micro {
                println!("[micro] Initializing hashmap for band {}", i);
            }
            hashmaps.push(HashMap::new());
        }

        let gamma = Gamma::new(2.0, 1.0).unwrap();
        let uniform = Uniform::new(0.0, 1.0).unwrap();

        let r_seeds: Vec<u32> = (0..num_bands).map(|i| seed + i as u32).collect();
        let c_seeds: Vec<u32> = (0..num_bands).map(|i| seed + i as u32 + num_bands as u32).collect();
        let b_seeds: Vec<u32> = (0..num_bands).map(|i| seed + i as u32 + num_bands as u32 * 2).collect();

        // Create a temporary instance to call hash_band
        let temp_lsh = WeightedMinHashLSH {
            hashmaps: hashmaps.clone(),
            num_bands,
            gamma: gamma.clone(),
            uniform: uniform.clone(),
            r_seeds: r_seeds.clone(),
            c_seeds: c_seeds.clone(),
            b_seeds: b_seeds.clone(),
            micro,
        };

        scored_documents.iter().enumerate().for_each(|(doc_id, vector)| {
            for band in 0..num_bands {
                let hash = temp_lsh.hash_band(vector.clone(), band);
                hashmaps[band].entry(hash).or_default().push(doc_id);
            }
        });

        temp_lsh
    }

    fn hash_band(&self, vector: Vec<(usize, f32)>, band: usize) -> usize {
        if self.micro {
            println!("[micro] WeightedMinHashLSH::hash called");
        }

        let t: Vec<(usize, f32)> = vector.iter().map(|&(index, weight)| {
            let r_ij = gamma_sample(index, band as u32, self.r_seeds[band], &self.gamma);
            let b_ij = uniform_sample(index, band as u32, self.b_seeds[band], &self.uniform);
            let t_ij = ((weight.ln() / r_ij) + b_ij).floor();
            (index, t_ij)
        }).collect();

        let y: Vec<(usize, f32)> = t.iter().map(|&(index, t_ij)| {
            let b_ij = uniform_sample(index, band as u32, self.b_seeds[band], &self.uniform);
            let r_ij = gamma_sample(index, band as u32, self.r_seeds[band], &self.gamma);

            let y_ij = (r_ij * (t_ij - b_ij)).exp();
            (index, y_ij)
        }).collect();

        let a: Vec<(usize, f32)> = y.iter().map(|&(index, y_ij)| {
            let c_ij = gamma_sample(index, band as u32, self.c_seeds[band], &self.gamma);
            let r_ij = gamma_sample(index, band as u32, self.r_seeds[band], &self.gamma);

            let a_ij = c_ij / (y_ij * r_ij.exp());
            (index, a_ij)
        }).collect();

        let min_index = a.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|&(index, _)| index);
        min_index.expect("At least one element should be present in the vector")
    }

    pub fn insert(self: &mut Self, vector: Vec<(usize, f32)>, doc_id: usize) {
        for band in 0..self.num_bands {
            let hash = self.hash_band(vector.clone(), band);
            self.hashmaps[band]
                .entry(hash)
                .or_default()
                .push(doc_id);
        }
    }

    pub fn query(self: &Self, vector: Vec<(usize, f32)>) -> Vec<usize> {
        if self.micro {
            println!("[micro] WeightedMinHashLSH::query called");
        }
        let mut results = Vec::new();
        for band in 0..self.num_bands {
            let hash_value = self.hash_band(vector.clone(), band);
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