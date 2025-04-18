#![allow(dead_code)]

use super::statistics::variance;
use super::structures::Sample;
use crate::tree::tree::{SplitParameters, StandardSplit};
use crate::RandomGenerator;
use core::f64;
use rand::Rng;
use rand::{seq::SliceRandom, SeedableRng};
use std::ops::Range;
use std::{
    cmp::{max, min},
    mem::swap,
};

pub fn train_test_split(
    data: &[Sample],
    test_size: f64,
    stratify: bool,
    random_state: Option<rand_chacha::ChaCha8Rng>,
) -> (Vec<Sample>, Vec<Sample>) {
    if data.len() < 2 && (test_size > 0. && test_size < 0.5) {
        panic!("The dataset is too small to be splitted.");
    }
    let mut indices: Vec<usize> = (0..data.len()).collect();
    let mut random_state =
        random_state.unwrap_or(rand_chacha::ChaCha8Rng::from_rng(rand::thread_rng()).unwrap());
    // Shuffle indices
    indices.shuffle(&mut random_state);

    let test_size = (data.len() as f64 * test_size) as usize;
    let test_size = min(data.len() - 1, max(1, test_size));

    let test_indices = &indices[..test_size];
    let train_indices = &indices[test_size..];

    let mut train_data: Vec<_> = train_indices.iter().map(|&i| data[i].clone()).collect();

    let mut test_data: Vec<_> = test_indices.iter().map(|&i| data[i].clone()).collect();

    if stratify {
        let mut count_train = train_data.iter().filter(|s| s.target == 0).count();
        let mut count_test = test_data.iter().filter(|s| s.target == 0).count();

        train_data.sort_by(|a, b| a.target.cmp(&b.target));
        test_data.sort_by(|a, b| a.target.cmp(&b.target));

        let mut idx = 1;
        while idx < test_size - 1 {
            let train_ratio = count_train as f64 / train_data.len() as f64;
            let test_ratio = count_test as f64 / test_data.len() as f64;
            if test_ratio > train_ratio {
                break;
            }
            let test_len = test_data.len();
            swap(&mut train_data[idx], &mut test_data[test_len - idx - 1]);
            count_train -= 1;
            count_test += 1;
            idx += 1;
        }
        let mut idx = 1;
        while idx < test_size - 1 {
            let train_ratio = count_train as f64 / train_data.len() as f64;
            let test_ratio = count_test as f64 / test_data.len() as f64;
            if test_ratio < train_ratio {
                break;
            }
            let train_len = train_data.len();
            swap(&mut train_data[train_len - idx - 1], &mut test_data[idx]);
            count_train += 1;
            count_test -= 1;
            idx += 1;
        }
    }
    train_data.shuffle(&mut random_state);
    test_data.shuffle(&mut random_state);
    (train_data, test_data)
}

fn split_samples<S: SplitParameters>(split: &S, samples: &mut [Sample]) -> usize {
    let mut start = 0;
    let mut end = samples.len();

    while start < end {
        if split.split(&samples[start]) == 0 {
            start += 1;
        } else {
            samples.swap(start, end - 1);
            end -= 1;
        }
    }
    start
}

pub fn get_random_split(
    samples: &mut [Sample],
    non_constant_features: &mut Vec<usize>,
    random_state: &mut RandomGenerator,
    min_samples_leaf: usize,
) -> Option<(Vec<Range<usize>>, StandardSplit, f64)> {
    non_constant_features.shuffle(random_state);

    while let Some(feature) = non_constant_features.pop() {
        let thresholds = samples
            .iter()
            .map(|f| f.features[feature])
            .collect::<Vec<_>>();

        let min_feature = *thresholds
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let max_feature = *thresholds
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        if max_feature - min_feature <= f64::EPSILON {
            // Remove constant features
            continue;
        } else {
            let threshold = random_state.gen_range(min_feature..max_feature);
            let rand_split = StandardSplit { feature, threshold };

            let split_idx = split_samples(&rand_split, samples);

            if split_idx < min_samples_leaf || (samples.len() - split_idx) < min_samples_leaf {
                continue;
            }

            non_constant_features.push(feature);

            return Some((
                vec![0..split_idx, split_idx..samples.len()],
                rand_split,
                f64::NAN,
            ));
        }
    }
    return None;
}

pub fn get_variance_split(
    samples: &mut [Sample],
    non_constant_features: &mut Vec<usize>,
    random_state: &mut RandomGenerator,
    min_samples_leaf: usize,
) -> Option<(Vec<Range<usize>>, StandardSplit, f64)> {
    let tries_count = ((samples[0].features.len() as f64).sqrt() as usize).max(1);

    let mut best_split = None;
    let mut best_variance = f64::INFINITY;

    for _ in 0..tries_count {
        if let Some((intervals, split, impurity)) = get_random_split(
            samples,
            non_constant_features,
            random_state,
            min_samples_leaf,
        ) {
            let values = samples
                .iter()
                .map(|s| s.features[split.feature])
                .collect::<Vec<_>>();

            let left_variance = (intervals[0].len() as f64 / samples.len() as f64)
                * variance(&values[intervals[0].clone()]);
            let right_variance = (intervals[1].len() as f64 / samples.len() as f64)
                * variance(&values[intervals[1].clone()]);

            let children_variance = left_variance + right_variance;
            if children_variance < best_variance {
                best_variance = children_variance;
                best_split = Some((intervals, split, impurity));
            }
        } else {
            break;
        }
    }
    best_split
}