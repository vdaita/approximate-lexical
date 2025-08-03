// based on the datasketch implementation of WeightedMinHash

use std::collections::HashMap;

use rand::prelude::*;use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use rand_distr::{Gamma, Uniform, Distribution};
use ndarray::{Array2, Array};
use ndarray::prelude::*;
use bitvec_simd::BitVec;

struct WeightedMinHash {
    pub seed: u64,
    pub hashvalues: Array2<f32>
}

struct WeightedMinHasher {
    pub num_perm: u32,
    pub seed: u64,
    pub discretization_param: u32,
    pub max_vocab_size: usize,
    pub hash_proj: Vec<Vec<bool>>
}

impl WeightedMinHasher {
    pub fn new(num_perm: u32, seed: u64, discretization_param: u32, max_vocab_size: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let hash_proj: Vec<Vec<bool>> = (0..num_perm)
            .map(|_| {
                (0..max_vocab_size)
                    .map(|_| rng.gen_bool(0.5)) // Randomly choose true/false
                    .collect()
            })
            .collect();

        Self {
            num_perm,
            seed,
            discretization_param,
            max_vocab_size,
            hash_proj
        }
    }

    fn round(&self, vector: Vec<(usize, f32)>) -> Vec<(usize, f32)> {
        let magnitude: f32 = vector.iter().map(|&(_, v)| v * v).sum::<f32>().sqrt();
        let unit_vector: Vec<(usize, f32)> = vector
            .iter()
            .map(|&(idx, val)| (idx, val / magnitude))
            .collect();
        
        let mut z_tilde: Vec<(usize, f32)> = unit_vector
            .iter()
            .map(|&(idx, val)| (idx, val.signum() * (floorf32(val * val * self.discretization_param) / self.discretization_param)))
            .collect();
        let i_star = z.iter()
            .max_by(|&(idx1, val1), &(idx2, val2)| (val1.abs()).partial_cmp(val2.abs()).unwrap())
            .map(|&(idx, _)| idx)
            .collect();
        
        let magnitude_z_tilde: f32 = z_tilde.iter().map(|&(_, v)| v * v).sum::<f32>().sqrt();
        let sigma = 1 - (magnitude_z_tilde * magnitude_z_tilde);

        z_tilde[i_star] = (i_star, sqrt(z_tilde[i_star].1 * z_tilde[i_star].1 + sigma));
        z_tilde
    }

    pub fn hash(&self, vector: Vec<(usize, f32)>) -> BitVec {
        let a_tilde = self.round(vector);
        
    }
}