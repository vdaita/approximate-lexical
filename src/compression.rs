pub fn compress_vector_alpha(vector: &[(usize, f32)], alpha: f32) -> Vec<(usize, f32)> {
    let mut sorted = vector.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut compressed = Vec::new();
    let mut total_weight = vector.iter().map(|&(_, weight)| weight).sum::<f32>();
    let mut current_weight = 0.0;
    for &(index, weight) in &sorted {
        if current_weight / total_weight < alpha {
            compressed.push((index, weight));
            current_weight += weight;
        } else {
            break;
        }
    }
    compressed
}

#[derive(Debug, Clone)]
pub struct DistributionQuantizer { 
    pub samples: Vec<f32>,
    pub ranges: Vec<(f32, f32)>,
    pub num_bins: usize
}

impl DistributionQuantizer {
    pub fn new(samples: Vec<f32>, num_bins: usize) -> Self {
        let min = *samples.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max = *samples.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let bin_size = (max - min) / num_bins as f32;
        
        let mut ranges = Vec::new();
        for i in 0..num_bins {
            let start = min + i as f32 * bin_size;
            let end = if i == num_bins - 1 { max } else { start + bin_size };
            ranges.push((start, end));
        }
        
        DistributionQuantizer { samples, ranges, num_bins }
    }

    pub fn quantize(&self, value: f32) -> usize {
        for (i, &(start, end)) in self.ranges.iter().enumerate() {
            if value >= start && value < end {
                return i;
            }
        }
        self.num_bins - 1 // Return the last bin if out of range
    }
}