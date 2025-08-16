use approximate_lexical::{build_index, query_index};
use bm25::{Embedder, EmbedderBuilder, Language, Scorer};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
