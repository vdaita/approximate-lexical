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
    pub hashindices: Array<usize>,
    pub hashvalues: Array<u32>
}

struct WeightedMinHasher {
    pub dim: u32,
    pub sample_size: usize,
    pub seed: usize,
    pub rs: Array2<f32>,
    pub ln_cs: Array2<f32>,
    pub betas: Array2<f32>,
}

impl WeightedMinHasher {
    pub fn new(dim: usize, sample_size: usize, seed: u64, micro: bool) -> Self {
        if micro {
            println!("[micro] WeightedMinHash::new called with dim={}, sample_size={}, seed={}", dim, sample_size, seed);  
        }
        
        let mut rng = rand::rng();

        let gamma_dist = Gamma::new(2.0, 1.0).unwrap();
        let uniform_dist = Uniform::new(0.0, 1.0);
        let rs = Array::from_shape_fn((sample_size, dim), |_| gamma_dist.sample(&mut rng) as f32);
        let ln_cs = Array.from_shape_fn((sample_size, dim), |_| {
            let sample = gamma_dist.sample(&mut rng);
            sample.ln() as f32
        });
        let betas = Array::from_shape_fn((sample_size, dim), |_| uniform_dist.sample(&mut rng) as f32);        

        Self {
            dim,
            sample_size,
            seed,
            rs,
            ln_cs,
            betas
        }
    }

    pub fn minhash(
        self: &Self,
        vector: Vec<(usize, f32)>,
        micro: bool
    ) -> WeightedMinHash {
        if micro {
            println!("[micro] WeightedMinHasher::minhash called with vector length={}", vector.len());
        }

        let mut hashindices = Array::<i32>::from_shape_fn(self.sample_size, |_| 0);
        let mut hashvalues = Array::<f32>::from_shape_fn(self.sample_size, |_| 0.0);
        let mut vlog: Vec<usize, f32> = vector.iter().map(|&(idx, val)| {
            return (idx, val.ln());
        }); // this seems to handle the normalization

        for sample_idx in 0..self.sample_size {
            let t: HashMap<usize, f32> = vector.iter().map(|&(vocab_idx,val)| {
                let hashed_vocab_idx = vocab_idx % self.dim;
                let t_val = (val / self.rs[[sample_idx, hashed_vocab_idx]]) + self.beta[[sample_idx, hashed_vocab_idx]];
                (*vocab_idx, t_val);
            }).collect();

            let ln_y: HashMap<usize, f32> = t.iter().map(|&(vocab_idx, t_val)| {
                let hashed_vocab_idx = vocab_idx % self.dim;
                (*vocab_idx, t_val - self.betas[[sample_idx, hashed_vocab_idx]]) * self.rs[[sample_idx, hashed_vocab_idx]]
            }).collect();

            let ln_a: Vec<(usize, f32)> = vector.iter().map(|(vocab_idx, _)| {
                let hashed_vocab_idx = vocab_idx % self.dim;
                (vocab_idx, self.ln_cs[[sample_idx, hashed_vocab_idx]] - ln_y.get(vocab_idx) - self.rs[[sample_idx, hashed_vocab_idx]])
            }).collect();

            let k = ln_a.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0);
            
            hashindices[sample_idx] = k;
            hashvalues[sample_idx] = t.get(k).floor() as u32;
        }

        WeightedMinMash {
            seed: self.seed,
            hashindices: hashindices,
            hashvalues: hashvalues
        }
    }

    pub fn get_bitstring_from_minhash(
        minhash: &WeightedMinHash,
        micro: bool
    ) -> BitVec {
        if micro {
            println!("[micro] WeightedMinHasher::get_bitstring_from_minhash called");
        }

        let mut bitstring = BitVec::with_capacity(minhash.hashindices.len());
        for &value in minhash.hashvalues.iter() {
            bitstring.push(value > 0);
        }
        bitstring
    }
}