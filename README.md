# TSRF-Dist: Reproducibility Section

This repository contains the reproducibility section for the paper:

**TSRF-Dist: A Novel Time Series Distance Based on Extremely Randomized Canonical Interval Forests**  
*Alberto Azzari, Manuele Bicego, Carlo Combi, Andrea Cracco, Pietro Sala*  
Published in *Data Mining and Knowledge Discovery*, 2025.  
DOI: [10.1007/s10618-025-01098-3](https://doi.org/10.1007/s10618-025-01098-3)

## Citation

If you use this repository, please cite the paper as follows:

```bibtex
@article{azzari2025,
    abstract = {This paper presents TSRF-Dist, a novel distance between time series based on Random Forests (RFs). We extend to the time-series domain concepts and tools of RF distances, a recent class of robust data-dependent distances defined for vectorial representations, thus proposing the first RF distance for time series. The distance is determined by (i) creating an RF to model a set of time series, and (ii) exploiting the trained RF to quantify the similarity between time series. As for the first step, we introduce in this paper the Extremely Randomized Canonical Interval Forest (ERCIF), a novel extension of Canonical Interval Forests that can model time series and can be trained without labels. We then exploit three different schemes, following ideas already employed in the vectorial case. The proposed distance, in different variants, has been thoroughly evaluated with 128 datasets from the UCR Time Series archive, showing promising results compared with literature alternatives.},
    author = {Azzari, Alberto and Bicego, Manuele and Combi, Carlo and Cracco, Andrea and Sala, Pietro},
    date = {2025/04/15},
    doi = {10.1007/s10618-025-01098-3},
    journal = {Data Mining and Knowledge Discovery},
    number = {3},
    pages = {27},
    title = {TSRF-Dist: a novel time series distance based on extremely randomized canonical interval forests},
    url = {https://doi.org/10.1007/s10618-025-01098-3},
    volume = {39},
    year = {2025}
}
```

## Usage

To compute the distances between the time series of the UCR archive, run the following command:

```bash
cargo test --release -- test_dmkd --nocapture
```

This computes the light version of the three distances implemented. To obtain the final results, you should also compute the heavy version, which uses 500 trees and the square root of the time series length as the number of intervals.

### Clustering
To obtain the final results, clustering must also be performed. We used the `kodama` library for agglomerative clustering. Below is the function used:

```rust
use hashbrown::HashMap;
use kodama::linkage;

pub fn agglomerative_clustering(
    n_clusters: usize,
    linkage_method: kodama::Method,
    distance_matrix: Vec<Vec<f64>>,
) -> Vec<isize> {
    let n = distance_matrix.len();
    let mut condensed_matrix = Vec::with_capacity((n * (n - 1)) / 2);
    for i in 0..n - 1 {
        for j in i + 1..n {
            condensed_matrix.push(distance_matrix[i][j]);
        }
    }
    let dendrogram = linkage(&mut condensed_matrix, n, linkage_method);
    let steps = dendrogram.steps();

    let mut clusters = HashMap::new();
    for i in 0..n {
        clusters.insert(i, vec![i]);
    }

    for (i, step) in steps.iter().enumerate() {
        if i >= n - n_clusters {
            break;
        }

        let (mut a, mut b) = (
            clusters.remove(&step.cluster1).unwrap(),
            clusters.remove(&step.cluster2).unwrap(),
        );

        let new_cluster = if a.len() > b.len() {
            a.extend(b);
            a
        } else {
            b.extend(a);
            b
        };

        clusters.insert(n + i, new_cluster);
    }
    assert_eq!(n_clusters, clusters.len());
    let mut labels = vec![-1; n];
    for (i, cluster) in clusters.iter() {
        for &j in cluster {
            labels[j] = *i as isize;
        }
    }
    labels
}
```

### Configuration

To change the path to the UCR archive, edit the file `tests/test_forusts` accordingly.

## Requirements

- Rust (latest stable version)
- UCR Time Series Archive datasets

## License

This repository is provided for academic purposes. Please refer to the paper for further details.