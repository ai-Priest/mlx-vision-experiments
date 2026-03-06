# MLX Vision Experiments

This repository contains benchmarks and experiments comparing MLX with PyTorch MPS on Apple Silicon.

## Milestone 1.1: Benchmarking Script

The `test_speed.py` script compares a simple 3-layer MLP between PyTorch MPS and MLX.

### Technical Decisions

- **MLX compilation (`@mx.compile`)**: We explicitly chose **not** to use `@mx.compile` for the training loop in this initial benchmark. Using `mx.compile` with stateful models appropriately requires `functools.partial` with exact state tracking (to prevent memory leaks or incorrect graphs), which adds complexity outside the scope of a basic benchmark. Eager MLX evaluation still leverages Metal GPU acceleration efficiently, offering a solid baseline comparison against PyTorch MPS.
- **Memory tracking**: We use native framework APIs (`torch.mps.current_allocated_memory()` and `mx.get_active_memory()`) to track GPU memory usage on Apple Silicon, as psutil and top only profile system RAM.

## Benchmark Results

| Metric | PyTorch MPS | Apple MLX |
|---|---|---|
| Training Time (per epoch) | 0.0611s | 0.0509s |
| Peak Memory (active) | 26.25 MB | 27.92 MB |

MLX is 16.7% faster per epoch than PyTorch MPS on M1 Max.
Memory usage is effectively equal between frameworks.

*Note: mx.get_active_memory() reports active memory at time of call, 
not true peak. Best available native metric without external profiling.*

## Hardware & Environment

| Config | Details |
|---|---|
| Machine | Apple M1 Max Mac Studio |
| Memory | 32GB Unified Memory |
| OS | macOS (March 2026) |
| Package Manager | Miniforge arm64 |
| MLX | 0.31.0 |
| PyTorch | MPS backend |

## Why MLX vs PyTorch MPS

CUDA benchmarks are irrelevant for Apple Silicon deployment. 
This repo benchmarks what actually matters for on-device Apple 
ML workloads: MLX versus PyTorch's own MPS backend, both running 
natively on the same unified memory architecture. Any Apple 
engineer can clone this repo and reproduce results on their 
own Mac in under 5 minutes.

## How to Reproduce

```bash
git clone https://github.com/ai-Priest/mlx-vision-experiments
cd mlx-vision-experiments
conda create -n apple_ai python=3.11
conda activate apple_ai
pip install mlx torch
python src/test_speed.py
```

Results print to terminal and save to benchmark_results/m1max_benchmark.md
