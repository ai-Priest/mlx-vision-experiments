import time
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from torchvision import datasets, transforms
from PIL import Image

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    data = dataset.data.numpy().astype(np.float32) / 255.0
    return data.reshape(-1, 784)

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)
        
    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        h = nn.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dims=[256, 512], output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        
    def __call__(self, z):
        h = nn.relu(self.fc1(z))
        h = nn.relu(self.fc2(h))
        # Use sigmoid to output probabilities (pixel values)
        return nn.sigmoid(self.fc3(h))

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim)
        
    def reparameterize(self, mu, logvar):
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(std.shape)
        return mu + eps * std
        
    def __call__(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def loss_fn(model, x):
    recon_x, mu, logvar = model(x)
    
    # Clip recon_x to prevent log(0) in BCE
    recon_x = mx.clip(recon_x, 1e-7, 1 - 1e-7)
    
    # Reconstruction loss: Binary Cross Entropy
    bce = -mx.sum(x * mx.log(recon_x) + (1 - x) * mx.log(1 - recon_x), axis=1)
    
    # KL Divergence
    kld = -0.5 * mx.sum(1 + logvar - mx.square(mu) - mx.exp(logvar), axis=1)
    
    return mx.mean(bce + kld)

def batch_iterate(batch_size, X):
    perm = np.random.permutation(len(X))
    for s in range(0, len(X), batch_size):
        ids = perm[s : s + batch_size]
        yield mx.array(X[ids])

def main():
    print("Loading MNIST dataset...")
    train_x = load_mnist()
    
    model = VAE()
    mx.eval(model.parameters())
    
    optimizer = optim.Adam(learning_rate=1e-3)
    
    # Implicitly compile-free, fully stateful implementation for MLX 0.31.0
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    epochs = 10
    batch_size = 128
    
    print("Starting training on natively supported Apple GPU via MLX (compile omitted)...")
    for epoch in range(epochs):
        start_time = time.perf_counter()
        
        model.train(True)
        total_loss = 0.0
        num_batches = 0
        
        for batch_x in batch_iterate(batch_size, train_x):
            loss, grads = loss_and_grad_fn(model, batch_x)
            optimizer.update(model, grads)
            
            # Evaluate loss, parameters, and optimizer state after updates
            mx.eval(loss, model.parameters(), optimizer.state)
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        end_time = time.perf_counter()
        epoch_time = end_time - start_time
        
        # Track memory footprint
        try:
            active_memory = mx.get_active_memory()
            mem_str = f"{active_memory / (1024**2):.2f} MB"
        except AttributeError:
            mem_str = "N/A"
            
        print(f"Epoch [{epoch+1}/{epochs}], Time: {epoch_time:.4f}s, Loss: {avg_loss:.4f}, Active Memory: {mem_str}")

    print("Generating samples...")
    # Sample 64 vectors
    z = mx.random.normal([64, 20])
    recon_x = model.decoder(z)
    mx.eval(recon_x)  # Force MLX lazy graph before numpy conversion
    
    # Convert to NumPy uint8
    samples = np.array(recon_x * 255).astype(np.uint8)
    
    # Stitch into an 8x8 grid
    samples = samples.reshape(64, 28, 28)
    grid = np.zeros((8 * 28, 8 * 28), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            grid[i*28:(i+1)*28, j*28:(j+1)*28] = samples[i*8+j]
            
    img = Image.fromarray(grid)
    save_path = "assets/generated_samples/vae_mlx_generated.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    print(f"Saved samples to {save_path}")

if __name__ == "__main__":
    main()
