use ndarray::{Array2};
use rand::rng;
use std::collections::HashMap;
use rand_distr::{Normal, Distribution};
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseToDense {
    vocab_map: HashMap<usize, usize>,
    random_matrix: Array2<f32>, // Now using f32 for Gaussian
}

impl SparseToDense {
    pub fn new(vocab_map: HashMap<usize, usize>, target_dim: usize) -> Self {
        let mut rng = rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let num_vocab = vocab_map.len();
        
        let random_matrix = Array2::from_shape_fn((num_vocab, target_dim), |_| {
            normal.sample(&mut rng) as f32
        });
        SparseToDense { vocab_map, random_matrix }
    }

    pub fn project(&self, vector: &Vec<(usize, f32)>) -> Vec<f32> {
        let target_dim = self.random_matrix.shape()[1];
        let mut result = vec![0.0; target_dim];
        for &(token_id, value) in vector {
            let token_random_idx = self.vocab_map.get(&token_id).cloned().unwrap_or(token_id);
            if token_id < self.random_matrix.shape()[0] {
                let row = self.random_matrix.row(token_random_idx);
                for (j, &r) in row.iter().enumerate() {
                    result[j] += value * r;
                }
            }
        }
        result
    }
    
    pub fn project_multiple(&self, vector: &Vec<Vec<(usize, f32)>>) -> Vec<Vec<f32>> {
        vector.par_iter().map(|v| self.project(v)).collect()
    }
}
