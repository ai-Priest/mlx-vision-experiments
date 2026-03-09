import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

def load_mnist_loader(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dims=[256, 512], output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def loss_fn(recon_x, x, mu, logvar):
    # Reconstruction loss: Binary Cross Entropy without reduction to sum over feature dim
    bce = F.binary_cross_entropy(recon_x, x, reduction='none').sum(dim=1)
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    # Mean over the batch
    return torch.mean(bce + kld)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 128
    epochs = 10
    
    print("Loading MNIST dataset...")
    train_loader = load_mnist_loader(batch_size=batch_size)
    
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting training PyTorch VAE on MPS...")
    for epoch in range(epochs):
        start_time = time.perf_counter()
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, _ in train_loader:
            # Flatten to 784 and move to device
            batch_x = batch_x.view(-1, 784).to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            
            loss = loss_fn(recon_x, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        
        # Crucial for accurate timing on MPS
        if device.type == 'mps':
            torch.mps.synchronize()
            
        end_time = time.perf_counter()
        epoch_time = end_time - start_time
        
        # Track memory footprint
        if device.type == 'mps':
            active_memory = torch.mps.driver_allocated_memory()
            mem_str = f"{active_memory / (1024**2):.2f} MB"
        else:
            mem_str = "N/A"
            
        print(f"Epoch [{epoch+1}/{epochs}], Time: {epoch_time:.4f}s, Loss: {avg_loss:.4f}, Active Memory: {mem_str}")

    print("Generating samples...")
    model.eval()
    with torch.no_grad():
        # Sample 64 vectors
        z = torch.randn(64, 20).to(device)
        recon_x = model.decoder(z)
        
        # Move back to CPU and convert to NumPy uint8
        samples = (recon_x.cpu().numpy() * 255).astype(np.uint8)
        
        # Stitch into an 8x8 grid
        samples = samples.reshape(64, 28, 28)
        grid = np.zeros((8 * 28, 8 * 28), dtype=np.uint8)
        for i in range(8):
            for j in range(8):
                grid[i*28:(i+1)*28, j*28:(j+1)*28] = samples[i*8+j]
                
        img = Image.fromarray(grid)
        save_path = "assets/generated_samples/vae_pytorch_mps_generated.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        print(f"Saved samples to {save_path}")

if __name__ == "__main__":
    main()
