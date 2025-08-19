use log::{debug, info, trace, warn};
use ndarray::Array2;
use rand::rng;
use rand_distr::{Distribution, Normal};
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseToDense {
    vocab_map: HashMap<usize, usize>,
    random_matrix: Array2<f32>, // Now using f32 for Gaussian
}

impl SparseToDense {
    pub fn new(vocab_list: &Vec<usize>, target_dim: usize) -> Self {
        let mut rng = rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let num_vocab = vocab_list.len();

        let random_matrix =
            Array2::from_shape_fn((num_vocab, target_dim), |_| normal.sample(&mut rng) as f32);

        let vocab_map: HashMap<usize, usize> = vocab_list
            .into_iter()
            .enumerate()
            .map(|(index, &term_id)| (term_id, index))
            .collect();

        SparseToDense {
            vocab_map,
            random_matrix,
        }
    }

    pub fn project(&self, vector: &Vec<(usize, f32)>) -> Vec<f32> {
        let target_dim = self.random_matrix.shape()[1];
        let mut result = vec![0.0; target_dim];

        for &(token_id, value) in vector {
            let token_random_idx = self.vocab_map.get(&token_id).cloned().unwrap_or(token_id);
            if token_random_idx < self.random_matrix.shape()[0] {
                let row = self.random_matrix.row(token_random_idx);
                for (j, &r) in row.iter().enumerate() {
                    result[j] += value * r;
                }
            } else {
                warn!(
                    "Token {} mapped to index {} which is out of bounds for matrix with {} rows",
                    token_id,
                    token_random_idx,
                    self.random_matrix.shape()[0]
                );
            }
        }
        trace!("Generated dense vector: {:?}", result);
        result
    }
}
