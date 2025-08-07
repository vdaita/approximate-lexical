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