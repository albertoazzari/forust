#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use std::fs;
    use crate::tree::transform::CACHE;
    use crate::utils::io::write_csv;
    use crate::utils::structures::MaxFeatures;
    use crate::{
        forest::{
            ciso_forest::CIsoForestConfig,
            erci_forest::ERCIForest,
            forest::{Forest, ForestConfig},
        },
        utils::{io::read_csv, structures::IntervalType},
    };

    #[test]
    fn test_dmkd() {
        // Settings for the experiments
        let config = CIsoForestConfig {
            n_intervals: IntervalType::LOG2,
            n_attributes: 8,
            outlier_config: ForestConfig {
                n_trees: 200,
                max_depth: Some(usize::MAX),
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_features: MaxFeatures::ALL,
                criterion: |_a, _b| 1.0,
                aggregation: None,
            },
        };
        let n_repetitions = 10;
        let paths = fs::read_dir("../../DATA/ucr").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
        let mut times = vec![vec![0.0; 4]; datasets.len()];
        for (i, path) in datasets.iter().enumerate() {

            println!("Processing dataset: {}", path.file_name().to_string_lossy());

            let ds_train = read_csv(
                path.path()
                    .join(format!("{}_TRAIN.tsv", path.file_name().to_string_lossy())),
                b'\t',
                false,
            )
            .unwrap();
            let ds_test = read_csv(
                path.path()
                    .join(format!("{}_TEST.tsv", path.file_name().to_string_lossy())),
                b'\t',
                false,
            )
            .unwrap();

            let mut ds = ds_train.clone();
            ds.extend(ds_test);

            for j in 0..n_repetitions {
                let mut model = ERCIForest::new(&config);
                let start_time = std::time::Instant::now();
                model.fit(
                    &mut ds,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                        ((i + 2) * (j + 2)) as u64,
                    )),
                );
                times[i][0] += start_time.elapsed().as_secs_f64();

                // breiman
                let start_time = std::time::Instant::now();
                let distance_matrix = model.pairwise_breiman(&ds, None);
                times[i][1] += start_time.elapsed().as_secs_f64();
                let breiman_path = format!(
                    "../../DATA/tsrf/LIGHT/breiman/{}_{}.csv",
                    path.file_name().to_string_lossy(),
                    j
                );
                write_csv(breiman_path, distance_matrix, None);

                // zhu
                let start_time = std::time::Instant::now();
                let distance_matrix = model.pairwise_zhu(&ds, None);
                times[i][2] += start_time.elapsed().as_secs_f64();
                let zhu_path = format!(
                    "../../DATA/tsrf/LIGHT/zhu/{}_{}.csv",
                    path.file_name().to_string_lossy(),
                    j
                );
                write_csv(zhu_path, distance_matrix, None);

                // ratiorf
                let start_time = std::time::Instant::now();
                let distance_matrix = model.pairwise_ratiorf(&ds, None);
                times[i][3] += start_time.elapsed().as_secs_f64();
                let ratiorf_path = format!(
                    "../../DATA/tsrf/LIGHT/ratiorf/{}_{}.csv",
                    path.file_name().to_string_lossy(),
                    j
                );
                write_csv(ratiorf_path, distance_matrix, None);
            }
            CACHE.clear();
            times[i][0] /= n_repetitions as f64;
            times[i][1] /= n_repetitions as f64;
            times[i][2] /= n_repetitions as f64;
            times[i][3] /= n_repetitions as f64;
            println!(
                "{}: Fit in {:.2}s, breiman in {:.2}s, zhu in {:.2}s, ratiorf in {:.2}s",
                path.file_name().to_string_lossy(),
                times[i][0],
                times[i][1],
                times[i][2],
                times[i][3],
            );
        }
        write_csv(
            "../../DATA/tsrf/LIGHT/times.csv",
            times,
            None,
        );
    }
}
