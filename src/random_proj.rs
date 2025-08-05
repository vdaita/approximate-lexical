use std::collections::HashMap;
use rand::rng;
use rand_distr::{Distribution, StandardNormal};
use crate::utils::{hamming_distance, hash_vector};

#[derive(Clone)]
pub struct RandomProjTermCluster {
    hash_proj: Vec<Vec<f32>>,
    hashes_to_documents: HashMap<u16, Vec<usize>>,
}

impl RandomProjTermCluster {
    pub fn new(documents: Vec<Vec<(usize, f32)>>, vocab_size: usize, micro: bool) -> Self {
        if micro {
            println!("[micro] TermCluster::new called with {} documents, vocab_size={}", documents.len(), vocab_size);
        }
        let hash_dim = 16;
        let mut rng = rng();
        let hash_proj: Vec<Vec<f32>> = (0..hash_dim)
            .map(|_| {
                (0..vocab_size)
                    .map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng))
                    .collect()
            })
            .collect();

        let mut hashes_to_documents: HashMap<u16, Vec<usize>> = HashMap::new();
        for (doc_idx, doc) in documents.iter().enumerate() {
            let hash = hash_vector(doc, &hash_proj);
            hashes_to_documents.entry(hash).or_default().push(doc_idx);
            if micro && doc_idx < 5 {
                println!("[micro] Document {} hashed to {}", doc_idx, hash);
            }
        }

        Self {
            hash_proj,
            hashes_to_documents,
        }
    }

    pub fn search(self: &mut Self, query: &[(usize, f32)], top_k: usize, micro: bool) -> Vec<usize> {
        if micro {
            println!("[micro] TermCluster::search called with top_k={}", top_k);
        }
        let query_hash = hash_vector(query, &self.hash_proj);
        if micro {
            println!("[micro] Query hash: {}", query_hash);
        }
        // BFS Hamming search
        let mut results = Vec::new();
        for dist in 0..=16 {
            if micro {
                println!("[micro] Hamming distance: {}", dist);
            }
            let mut found = 0;
            for hash in (0..=u16::MAX).filter(|h: &u16| hamming_distance(*h, query_hash) == dist) {
                if let Some(docs) = self.hashes_to_documents.get(&hash) {
                    results.extend(docs);
                    found += docs.len();
                    if micro {
                        println!("[micro] Found {} docs at hash {} (dist {})", docs.len(), hash, dist);
                    }
                    if results.len() >= top_k {
                        if micro {
                            println!("[micro] Enough results found, returning.");
                        }
                        return results.into_iter().take(top_k).collect();
                    }
                }
            }
            if micro {
                println!("[micro] Total found at dist {}: {}", dist, found);
            }
        }
        if micro {
            println!("[micro] Finished search, returning {} results.", results.len());
        }
        results.into_iter().take(top_k).collect()
    }
}
