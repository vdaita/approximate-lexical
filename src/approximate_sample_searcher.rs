use std::collections::{HashMap};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rand::distr::weighted::WeightedIndex;
use rand::thread_rng;
use rand::distr::Distribution;

