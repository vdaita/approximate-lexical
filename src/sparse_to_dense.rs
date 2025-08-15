use ndarray::{Array2, Axis};
use rand::thread_rng;
use rand_distr::{Normal, Distribution};
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;

pub struct SparseToDense {
    random_matrix: Array2<f32>, // Now using f32 for Gaussian
}

impl SparseToDense {
    pub fn new(max_vocab_size: usize, target_dim: usize) -> Self {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let random_matrix = Array2::from_shape_fn((max_vocab_size, target_dim), |_| {
            normal.sample(&mut rng) as f32
        });
        SparseToDense { random_matrix }
    }

    pub fn project(&self, vector: &Vec<(usize, f32)>) -> Vec<f32> {
        let target_dim = self.random_matrix.shape()[1];
        let mut result = vec![0.0; target_dim];
        for &(token_id, value) in vector {
            if token_id < self.random_matrix.shape()[0] {
                let row = self.random_matrix.row(token_id);
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
