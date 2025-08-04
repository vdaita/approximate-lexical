// based on the datasketch implementation of LSHForest
use crate::weighted_minhash::{WeightedMinHash, WeightedMinHasher};

struct LSH {
    pub seed: u64,
    pub threshold: f32,
    pub num_permutations: usize,
    pub num_bands: usize,

    pub ranges: Vec<(usize, usize)>,
    pub hash_tables: Vec<HashMap<u32, Vec<usize>>>,
    pub micro: bool
}

impl LSH {
    pub fn new(seed: u64, threshold: f32, num_permutations: usize, num_bands: usize, band_range: usize, micro: bool) -> Self {
        println!("[micro] LSH::new called with seed={}, threshold={}, num_permutations={}, num_bands={}, band_range={}", seed, threshold, num_permutations, num_bands, band_range);

        let mut ranges = Vec::with_capacity(num_bands);
        let band_range = num_permutations / num_bands;
        for i in 0..num_bands {
            let start = i * band_range;
            let end = if i == num_bands - 1 {
                num_permutations
            } else {
                start + band_range
            };
            ranges.push((start, end));
        }

        let hash_tables = vec![HashMap::new(); num_bands];

        Self {
            seed,
            threshold,
            num_permutations,
            num_bands,
            hash_tables,
            ranges,
            micro
        }
    }

    pub fn generate(&self, minhash: &WeightedMinHash) -> Vec<u16> {
        
    }
}