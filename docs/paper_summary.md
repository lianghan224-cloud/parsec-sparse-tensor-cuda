\# PARSEC Paper Summary



\## Problem



Large-scale sparse tensor completion suffers from two major system-level bottlenecks:



1\. Highly uneven observation distribution causes load imbalance.

2\. Static scheduling wastes computation on low-yield blocks during later training stages.



\## Method



PARSEC combines:



\- Recursive load-balanced partitioning

\- Block-level dynamic freezing

\- Error-based reactivation

\- CP-SGD optimization



\## My Contributions



I mainly contributed to:



\- Experiment exploration

\- CUDA/Python implementation support

\- Runtime and RMSE evaluation

\- Ablation result analysis

\- Efficiency-accuracy trade-off analysis



\## Key Result



On BJTaxi with sampling rate 0.5:



\- Only divide runtime: 12.15 s

\- PARSEC runtime: 8.57 s

\- Speedup: 1.42×

\- RMSE changes from 0.038928 to 0.042199



\## Relevance to AI Infrastructure



The project is related to AI infrastructure and LLM inference optimization from the following perspectives:



\- GPU-friendly workload partitioning

\- Dynamic scheduling

\- Redundant computation reduction

\- Efficiency-accuracy trade-off

\- Freeze/reactivation mechanism



These ideas can be extended to LLM KV Cache management, where historical KV segments may be kept dense, compressed, frozen, or restored according to importance.

