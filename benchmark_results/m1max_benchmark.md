# M1 Max Benchmark: PyTorch MPS vs MLX

| Framework     | Time per Epoch (s) | Peak Memory (MB) |
|--------------|--------------------|------------------|
| PyTorch MPS  | 0.0611             | 26.25            |
| MLX          | 0.0509             | 27.92            |

*Benchmarked on Apple M1 Max Mac Studio using a simple 3-layer MLP.*

## VAE Training — Apple MLX vs PyTorch MPS (MNIST)

| Metric                        | Apple MLX       |
|-------------------------------|-----------------|
| Avg time per epoch            | 0.902s          |
| Active memory (live arrays)   | 25.29 MB        |
| Loss at epoch 10              | 119.09          |
| Architecture                  | 784→512→256→z20 |
| Dataset                       | MNIST (60,000)  |
| Batch size                    | 128             |

## VAE Benchmark — MNIST (M1 Max Mac Studio)

| Metric                  | Apple MLX   | PyTorch MPS | Winner        |
|-------------------------|-------------|-------------|---------------|
| Avg time per epoch      | 0.902s      | 3.460s      | MLX ✅ 3.84x faster |
| Memory reading          | 25.29 MB    | 64.66 MB    | Not comparable†     |
| Loss at epoch 10        | 119.09      | 103.68      | PyTorch 13% lower   |
| Metal warmup (epoch 1)  | 1.565s      | 4.849s      | MLX ✅              |

**Architecture:** 784 → 512 → 256 → z(20) → 256 → 512 → 784  
**Dataset:** MNIST 60,000 samples | Batch size: 128 | Epochs: 10  
**Optimizer:** Adam lr=1e-3  

> On identical VAE architecture on Apple M1 Max, the MLX script trains
> 3.84x faster end to end than the PyTorch MPS script. The data pipelines
> differ, so part of the gap is data loading, not the framework.
>
> † MLX reports `mx.get_active_memory()` (live arrays); PyTorch reports
> `torch.mps.driver_allocated_memory()` (whole driver pool). Different measures.
```

---


``` Milestone 1.2 complete - MLX 3.84x faster than PyTorch MPS on VAE benchmark
