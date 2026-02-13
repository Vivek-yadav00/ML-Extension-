# Complexity Mapping Rules

This document defines how ML models are mapped to Big-O complexity.

## Scikit-Learn

| Model | Time Complexity | Memory Complexity | Notes |
|-------|----------------|-------------------|-------|
| SVC | O(n_samples^2 * n_features) | O(n_samples * n_features) | Slow on large datasets |
| RandomForest | O(n_trees * n_samples * log(n_samples)) | O(n_trees * n_leaves) | Highly parallelizable |
| KMeans | O(n_samples * k * i) | O(n_samples * n_features) | Linear w.r.t data size |

## Deep Learning (Heuristic)

| Layer Type | Complexity Factor |
|------------|-------------------|
| Dense (FC) | O(input * output) |
| Conv2D | O(k^2 * input * output * H * W) |
| RNN/LSTM | O(time * weights) |

*Note: Deep learning complexity varies significantly based on architecture depth.*
