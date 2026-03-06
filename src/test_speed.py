import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlx.core as mx
import mlx.nn as mx_nn
import mlx.optimizers as mx_optim
from typing import Tuple

# Parameters
BATCH_SIZE = 128
INPUT_DIM = 512
HIDDEN_DIM = 256
OUTPUT_DIM = 10
NUM_EPOCHS = 20
NUM_BATCHES = 100

class PyTorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class MLXMLP(mx_nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = mx_nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = mx_nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def __call__(self, x):
        return self.fc2(mx.maximum(self.fc1(x), 0.0))

def run_pytorch_benchmark() -> Tuple[float, float]:
    device = torch.device('mps')
    model = PyTorchMLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Create dummy data
    X = torch.randn(NUM_BATCHES, BATCH_SIZE, INPUT_DIM, device=device)
    y = torch.randint(0, OUTPUT_DIM, (NUM_BATCHES, BATCH_SIZE), device=device)

    # Warmup
    for _ in range(2):
        out = model(X[0])
        loss = criterion(out, y[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.mps.synchronize()
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        for i in range(NUM_BATCHES):
            optimizer.zero_grad()
            out = model(X[i])
            loss = criterion(out, y[i])
            loss.backward()
            optimizer.step()
    
    torch.mps.synchronize()
    end_time = time.time()
    
    time_per_epoch = (end_time - start_time) / NUM_EPOCHS
    peak_mem = torch.mps.current_allocated_memory() / (1024 ** 2) # Processed in MB
    
    # Cleanup memory
    del model, optimizer, X, y
    torch.mps.empty_cache()
    
    return time_per_epoch, peak_mem

def mlx_loss_fn(model, X, y):
    logits = model(X)
    loss = mx.mean(mx_nn.losses.cross_entropy(logits, y))
    return loss

def run_mlx_benchmark() -> Tuple[float, float]:
    model = MLXMLP()
    mx.eval(model.parameters())
    
    optimizer = mx_optim.SGD(learning_rate=0.01)
    
    loss_and_grad_fn = mx_nn.value_and_grad(model, mlx_loss_fn)
    
    def step(X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss
        
    # Create dummy data
    X_batches = [mx.random.normal((BATCH_SIZE, INPUT_DIM)) for _ in range(NUM_BATCHES)]
    y_batches = [mx.random.randint(0, OUTPUT_DIM, (BATCH_SIZE,)) for _ in range(NUM_BATCHES)]
    
    mx.eval(X_batches, y_batches)

    # Warmup
    for _ in range(2):
        loss = step(X_batches[0], y_batches[0])
        mx.eval(model.parameters(), optimizer.state, loss)
    
    mx.eval(model.parameters())
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        for i in range(NUM_BATCHES):
            loss = step(X_batches[i], y_batches[i])
            mx.eval(model.parameters(), optimizer.state, loss)
    
    # Ensure execution is complete before stopping timer
    mx.eval(model.parameters())
    end_time = time.time()
    
    time_per_epoch = (end_time - start_time) / NUM_EPOCHS

    # Note: get_active_memory() returns current active memory, not true peak.
    # Best available native metric in MLX without external profiling tools.
    peak_mem = mx.get_active_memory() / (1024 ** 2) # Processed in MB
    
    return time_per_epoch, peak_mem

def main():
    print("Running PyTorch MPS Benchmark...")
    pt_time, pt_mem = run_pytorch_benchmark()
    print("Running MLX Benchmark...")
    mlx_time, mlx_mem = run_mlx_benchmark()

    # Markdown Summary
    md_content = f"""# M1 Max Benchmark: PyTorch MPS vs MLX

| Framework     | Time per Epoch (s) | Peak Memory (MB) |
|--------------|--------------------|------------------|
| PyTorch MPS  | {pt_time:.4f}             | {pt_mem:.2f}            |
| MLX          | {mlx_time:.4f}             | {mlx_mem:.2f}            |

*Benchmarked on Apple M1 Max Mac Studio using a simple 3-layer MLP.*
"""
    
    print("\nBenchmark Summary:")
    print("-" * 65)
    print(f"{'Framework':<15} | {'Time per Epoch (s)':<20} | {'Peak Memory (MB)':<20}")
    print("-" * 65)
    print(f"{'PyTorch MPS':<15} | {pt_time:<20.4f} | {pt_mem:<20.2f}")
    print(f"{'MLX':<15} | {mlx_time:<20.4f} | {mlx_mem:<20.2f}")
    print("-" * 65)

    res_dir = os.path.expanduser(
        "~/AI_Dojo/Projects/mlx-vision-experiments/benchmark_results"
    )
    os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, "m1max_benchmark.md")
    with open(out_file, "w") as f:
        f.write(md_content)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()