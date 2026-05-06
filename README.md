# PARSEC: Adaptive Parallel Sparse Tensor Completion with Dynamic Reactivation

This repository organizes my implementation notes, experiment logs, and reproducibility materials for the paper:

Adaptive Parallel Sparse Tensor Completion with Load-Balanced Partitioning and Dynamic Reactivation

I am the second author of this work. I mainly contributed to experiment exploration, CUDA-related implementation, evaluation scripts, and result analysis.

## 1. Background

Sparse tensor completion aims to recover missing entries from partially observed high-dimensional data. It is useful in scenarios such as network traffic monitoring, spatiotemporal analysis, anomaly detection, and incomplete data recovery.

For a third-order sparse tensor X with shape I x J x K, CP decomposition approximates each observed entry by factor matrices A, B, and C.

The reconstruction of an observed entry can be written as:

    x_hat[i, j, k] = sum over r of A[i, r] * B[j, r] * C[k, r]

The factor matrices are optimized only on observed entries using stochastic gradient descent.

## 2. Motivation

In large-scale sparse tensor completion, observed entries are often highly unevenly distributed. Regular partitioning may cause severe workload imbalance on GPU or parallel execution units.

This project focuses on two system-level problems:

1. Load imbalance caused by uneven observation distribution.
2. Redundant computation caused by updating all blocks uniformly during all training stages.

## 3. Method Overview

PARSEC introduces two main designs:

### 3.1 Load-Balanced Recursive Partitioning

The sparse tensor space is recursively divided according to the distribution of observed entries.

Dense regions are further split, while sparse regions remain coarse. Each leaf node is treated as a parallel execution block.

This design aims to make the workload of different blocks more balanced and reduce idle time caused by uneven task distribution.

### 3.2 Dynamic Freezing and Reactivation

During SGD optimization, each block tracks its local reconstruction error improvement.

If a block becomes stable for several consecutive iterations, it is temporarily frozen to avoid low-yield updates.

If the reconstruction error of a frozen block increases later, the block is reactivated and inserted back into the update queue.

This mechanism reduces redundant computation while keeping the model able to correct degraded regions.

## 4. My Contributions

My main contributions include:

- Implemented and tested the sparse tensor CP-SGD workflow.
- Participated in the design and validation of block-level freezing and reactivation.
- Built evaluation scripts for runtime and RMSE comparison.
- Conducted experiments on real-world datasets, including BJTaxi, PeMS, ECW08, and CBW.
- Analyzed the efficiency-accuracy trade-off introduced by dynamic reactivation.
- Organized experiment logs and result tables for paper writing and ablation analysis.

## 5. Key Result

Under a sampling rate of 0.5 on the BJTaxi dataset, the dynamic reactivation mechanism reduces cumulative GPU runtime while introducing a moderate RMSE increase.

| Method | Cumulative GPU Runtime | Final RMSE | Speedup |
|---|---:|---:|---:|
| Only divide | 12.15 s | 0.038928 | 1.00x |
| PARSEC | 8.57 s | 0.042199 | 1.42x |

The speedup is calculated as:

    12.15 / 8.57 = 1.42x

This result shows that PARSEC can reduce redundant updates and improve training efficiency, while still maintaining a controllable reconstruction error.

## 6. Efficiency-Accuracy Trade-off

The dynamic reactivation mechanism improves runtime efficiency by freezing blocks with low update benefits.

However, freezing blocks may slightly increase reconstruction error because some local updates are skipped.

Therefore, this project studies the trade-off among:

- GPU runtime
- Reconstruction accuracy
- Block scheduling overhead
- Redundant computation reduction

This trade-off is important for large-scale AI infrastructure systems, where both efficiency and accuracy matter.

## 7. Relevance to AI Infrastructure

Although this work focuses on sparse tensor completion, the system ideas are closely related to AI infrastructure and LLM inference optimization.

The following ideas can be transferred to long-context LLM systems:

- Workload-aware partitioning
- GPU-friendly task organization
- Dynamic scheduling
- Freezing and reactivation
- Efficiency-accuracy trade-off
- Segment-level state management

For example, in LLM KV Cache management, historical KV segments may be divided into different states:

- Recent segments remain dense.
- Important historical segments remain active.
- Low-importance segments may be compressed or frozen.
- Frozen segments may be restored when their importance increases.

This connection motivates my ongoing work on segmented KV Cache compression for long-context LLM inference.

## 8. Repository Structure

The repository is organized as follows:

    parsec-sparse-tensor-cuda/
    ├── src/        Core implementation or code snippets
    ├── scripts/    Experiment and plotting scripts
    ├── results/    Runtime and RMSE result tables
    ├── docs/       Paper summary and technical notes
    └── README.md   Project overview

## 9. Current Status

Current repository status:

- Initial project documentation has been added.
- Initial ablation result tables have been added.
- Paper summary notes have been added.
- Runnable experiment scripts are being cleaned and will be added later.

## 10. TODO

- [ ] Clean and upload runnable experiment scripts.
- [ ] Add dataset preprocessing instructions.
- [ ] Add runtime and RMSE plotting scripts.
- [ ] Add CUDA implementation notes.
- [ ] Add reproducibility commands.
- [ ] Add more detailed performance analysis.
- [ ] Connect the system design idea to segmented LLM KV Cache compression.
