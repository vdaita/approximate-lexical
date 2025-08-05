pub fn hamming_distance(a: u16, b: u16) -> u32 {
    (a ^ b).count_ones()
}

pub fn hash_vector(vector: &[(usize, f32)], hash_proj: &[Vec<f32>]) -> u16 {
    let mut hash: u16 = 0;
    for (i, proj) in hash_proj.iter().enumerate() {
        let mut score = 0.0;
        for (idx, val) in vector.iter() {
            score += proj[idx % proj.len()] * val;
        }
        if score > 0.0 {
            hash |= 1 << i;
        }
    }
    hash
}