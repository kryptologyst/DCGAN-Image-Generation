"""
Minimal DCGAN Implementation
==========================

A simplified version of the DCGAN to avoid system-level issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
from pathlib import Path

class SimpleConfig:
    """Simple configuration class."""
    def __init__(self):
        self.latent_dim = 100
        self.image_size = 64
        self.channels = 1  # MNIST is grayscale
        self.generator_features = 64
        self.discriminator_features = 64
        self.batch_size = 64
        self.num_epochs = 5
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.dataset = "mnist"
        self.data_path = "./data"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42

class SimpleGenerator(nn.Module):
    """Simple DCGAN Generator."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        latent_dim = config.latent_dim
        ngf = config.generator_features
        channels = config.channels
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

class SimpleDiscriminator(nn.Module):
    """Simple DCGAN Discriminator."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        ndf = config.discriminator_features
        channels = config.channels
        
        self.main = nn.Sequential(
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def create_dataset(config):
    """Create dataset with transforms."""
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = MNIST(root=config.data_path, train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    return dataset, dataloader

def train_simple_dcgan():
    """Train a simple DCGAN."""
    print("🎨 Starting Simple DCGAN Training...")
    
    # Setup
    config = SimpleConfig()
    torch.manual_seed(config.seed)
    
    # Create models
    generator = SimpleGenerator(config).to(config.device)
    discriminator = SimpleDiscriminator(config).to(config.device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    
    # Dataset
    dataset, dataloader = create_dataset(config)
    
    # Training loop
    for epoch in range(config.num_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(config.device)
            batch_size = real_imgs.size(0)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real images
            real_validity = discriminator(real_imgs)
            real_loss = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity))
            
            # Fake images
            z = torch.randn(batch_size, config.latent_dim, 1, 1, device=config.device)
            fake_imgs = generator(z).detach()
            fake_validity = discriminator(fake_imgs)
            fake_loss = F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, config.latent_dim, 1, 1, device=config.device)
            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs)
            g_loss = F.binary_cross_entropy(fake_validity, torch.ones_like(fake_validity))
            
            g_loss.backward()
            optimizer_G.step()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{config.num_epochs} | Batch {i} | D_Loss: {d_loss:.4f} | G_Loss: {g_loss:.4f}")
    
    print("✅ Training completed!")
    return generator, discriminator, config

def generate_images(generator, config, num_images=16):
    """Generate images."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, config.latent_dim, 1, 1, device=config.device)
        fake_imgs = generator(z)
        return fake_imgs.cpu()

def main():
    """Main function."""
    print("🎨 Simple DCGAN Demo")
    print("=" * 30)
    
    try:
        # Train the model
        generator, discriminator, config = train_simple_dcgan()
        
        # Generate images
        print("🎨 Generating sample images...")
        images = generate_images(generator, config, 16)
        
        # Save images as numpy array
        images_np = images.numpy()
        np.save("simple_generated_images.npy", images_np)
        
        print(f"✅ Generated {images_np.shape[0]} images")
        print(f"✅ Image shape: {images_np.shape[1:]} (channels, height, width)")
        print(f"✅ Value range: [{images_np.min():.3f}, {images_np.max():.3f}]")
        print("✅ Images saved as 'simple_generated_images.npy'")
        
        # Model info
        print(f"\n📊 Model Statistics:")
        print(f"  Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
        print(f"  Total parameters: {sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters()):,}")
        
        print("\n🎉 Simple DCGAN demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
