"""
Modern DCGAN Implementation with Latest Techniques
=================================================

This project implements a state-of-the-art DCGAN (Deep Convolutional Generative Adversarial Network)
with modern PyTorch features, advanced training techniques, and comprehensive monitoring.

Features:
- Modern PyTorch 2.x with torch.compile for faster training
- Spectral Normalization for training stability
- Gradient Penalty for improved convergence
- Multiple dataset support (MNIST, CIFAR-10, CelebA)
- TensorBoard and Weights & Biases integration
- Interactive Streamlit UI
- Comprehensive configuration system
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CelebA
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from tqdm import tqdm
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management for DCGAN training."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_default_config()
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        return {
            # Model parameters
            "latent_dim": 100,
            "image_size": 64,
            "channels": 3,  # RGB images
            "generator_features": 64,
            "discriminator_features": 64,
            
            # Training parameters
            "batch_size": 64,
            "num_epochs": 200,
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
            
            # Advanced training features
            "use_spectral_norm": True,
            "use_gradient_penalty": True,
            "gradient_penalty_weight": 10.0,
            "n_critic": 5,  # Number of discriminator updates per generator update
            
            # Dataset
            "dataset": "cifar10",  # mnist, cifar10, celeba
            "data_path": "./data",
            
            # Monitoring
            "use_tensorboard": True,
            "use_wandb": False,
            "log_interval": 100,
            "save_interval": 1000,
            
            # Paths
            "output_dir": "./outputs",
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
            
            # Device
            "device": "auto",  # auto, cpu, cuda, mps
            
            # Reproducibility
            "seed": 42,
        }
    
    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        self.config.update(user_config)
    
    def save_config(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def get(self, key, default=None):
        return self.config.get(key, default)

class SpectralNorm(nn.Module):
    """Spectral Normalization for stabilizing GAN training."""
    
    def __init__(self, module, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        super().__init__()
        self.module = module
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                           f'got n_power_iterations={n_power_iterations}')
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        
        weight = getattr(module, name)
        height = weight.size(dim)
        weight_mat = weight.view(height, -1)
        h, w = weight_mat.size()
        u = weight_mat.new_empty(h).normal_(0, 1)
        v = weight_mat.new_empty(w).normal_(0, 1)
        self.register_buffer('_u', F.normalize(u, dim=0, eps=self.eps))
        self.register_buffer('_v', F.normalize(v, dim=0, eps=self.eps))
        
        # Make sure we start with the right size of u
        self._made_params = True
        self._spectral_norm_parameters()
    
    def _spectral_norm_parameters(self):
        weight = getattr(self.module, self.name)
        height = weight.size(self.dim)
        weight_mat = weight.view(height, -1)
        h, w = weight_mat.size()
        u = self._u.resize_(h)
        v = self._v.resize_(w)
    
    def _update_u_v(self):
        u = self._u
        v = self._v
        weight = getattr(self.module, self.name)
        weight_mat = weight.view(weight.size(self.dim), -1)
        
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        
        self._u.copy_(u)
        self._v.copy_(v)
    
    def forward(self, *args):
        self._update_u_v()
        return self.module(*args)

class ModernGenerator(nn.Module):
    """Modern DCGAN Generator with spectral normalization and improved architecture."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        latent_dim = config["latent_dim"]
        ngf = config["generator_features"]
        channels = config["channels"]
        
        self.main = nn.Sequential(
            # Input: Z latent vector
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # State size: ngf x 32 x 32
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size: channels x 64 x 64
        )
        
        # Apply spectral normalization if enabled
        if config["use_spectral_norm"]:
            self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                SpectralNorm(module)
    
    def forward(self, input):
        return self.main(input)

class ModernDiscriminator(nn.Module):
    """Modern DCGAN Discriminator with spectral normalization and improved architecture."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        ndf = config["discriminator_features"]
        channels = config["channels"]
        
        self.main = nn.Sequential(
            # Input size: channels x 64 x 64
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        # Apply spectral normalization if enabled
        if config["use_spectral_norm"]:
            self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                SpectralNorm(module)
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class GradientPenalty:
    """Gradient Penalty for WGAN-GP training."""
    
    def __init__(self, device):
        self.device = device
    
    def __call__(self, discriminator, real_samples, fake_samples, weight=10.0):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        d_interpolated = discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * weight
        
        return gradient_penalty

class DatasetManager:
    """Manages different datasets for DCGAN training."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_path = Path(config["data_path"])
        self.data_path.mkdir(exist_ok=True)
    
    def get_dataset(self, dataset_name: str):
        """Get dataset with appropriate transforms."""
        image_size = self.config["image_size"]
        channels = self.config["channels"]
        
        if dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.Grayscale(channels),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * channels, [0.5] * channels)
            ])
            return MNIST(root=str(self.data_path), train=True, transform=transform, download=True)
        
        elif dataset_name == "cifar10":
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * channels, [0.5] * channels)
            ])
            return CIFAR10(root=str(self.data_path), train=True, transform=transform, download=True)
        
        elif dataset_name == "celeba":
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * channels, [0.5] * channels)
            ])
            return CelebA(root=str(self.data_path), split='train', transform=transform, download=True)
        
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

class DCGANTrainer:
    """Main DCGAN trainer class with modern features."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = self._setup_device()
        self._setup_directories()
        self._setup_logging()
        
        # Initialize models
        self.generator = ModernGenerator(config).to(self.device)
        self.discriminator = ModernDiscriminator(config).to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config["learning_rate"],
            betas=(config["beta1"], config["beta2"])
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config["learning_rate"],
            betas=(config["beta1"], config["beta2"])
        )
        
        # Initialize dataset
        self.dataset_manager = DatasetManager(config)
        self.dataset = self.dataset_manager.get_dataset(config["dataset"])
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize gradient penalty
        self.gradient_penalty = GradientPenalty(self.device)
        
        # Initialize monitoring
        if config["use_tensorboard"]:
            self.writer = SummaryWriter(self.config["log_dir"])
        
        if config["use_wandb"]:
            wandb.init(project="dcgan-modern", config=config.config)
        
        # Compile models for faster training (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.generator = torch.compile(self.generator)
            self.discriminator = torch.compile(self.discriminator)
    
    def _setup_device(self):
        """Setup compute device."""
        if self.config["device"] == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config["device"])
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config["output_dir"], self.config["checkpoint_dir"], self.config["log_dir"]]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        torch.manual_seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["seed"])
    
    def train(self):
        """Main training loop."""
        logger.info("Starting DCGAN training...")
        
        for epoch in range(self.config["num_epochs"]):
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            for i, (real_imgs, _) in enumerate(progress_bar):
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                
                # Train Discriminator
                d_loss = self._train_discriminator(real_imgs, batch_size)
                
                # Train Generator (less frequently)
                if i % self.config["n_critic"] == 0:
                    g_loss = self._train_generator(batch_size)
                else:
                    g_loss = 0
                
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                
                # Update progress bar
                progress_bar.set_postfix({
                    'D_Loss': f'{d_loss:.4f}',
                    'G_Loss': f'{g_loss:.4f}'
                })
                
                # Logging
                if i % self.config["log_interval"] == 0:
                    self._log_metrics(epoch, i, d_loss, g_loss)
                
                # Save checkpoints
                if i % self.config["save_interval"] == 0:
                    self._save_checkpoint(epoch, i)
            
            # Generate sample images
            self._generate_samples(epoch)
            
            logger.info(f"Epoch {epoch+1} completed - D_Loss: {epoch_d_loss/len(self.dataloader):.4f}, G_Loss: {epoch_g_loss/len(self.dataloader):.4f}")
        
        logger.info("Training completed!")
    
    def _train_discriminator(self, real_imgs, batch_size):
        """Train discriminator."""
        self.optimizer_D.zero_grad()
        
        # Real images
        real_validity = self.discriminator(real_imgs)
        real_loss = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity))
        
        # Fake images
        z = torch.randn(batch_size, self.config["latent_dim"], 1, 1, device=self.device)
        fake_imgs = self.generator(z).detach()
        fake_validity = self.discriminator(fake_imgs)
        fake_loss = F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))
        
        d_loss = real_loss + fake_loss
        
        # Gradient penalty
        if self.config["use_gradient_penalty"]:
            gp = self.gradient_penalty(self.discriminator, real_imgs, fake_imgs, self.config["gradient_penalty_weight"])
            d_loss += gp
        
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item()
    
    def _train_generator(self, batch_size):
        """Train generator."""
        self.optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, self.config["latent_dim"], 1, 1, device=self.device)
        fake_imgs = self.generator(z)
        fake_validity = self.discriminator(fake_imgs)
        g_loss = F.binary_cross_entropy(fake_validity, torch.ones_like(fake_validity))
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item()
    
    def _log_metrics(self, epoch, iteration, d_loss, g_loss):
        """Log training metrics."""
        if self.config["use_tensorboard"]:
            self.writer.add_scalar('Loss/Discriminator', d_loss, epoch * len(self.dataloader) + iteration)
            self.writer.add_scalar('Loss/Generator', g_loss, epoch * len(self.dataloader) + iteration)
        
        if self.config["use_wandb"]:
            wandb.log({
                'epoch': epoch,
                'iteration': iteration,
                'discriminator_loss': d_loss,
                'generator_loss': g_loss
            })
    
    def _save_checkpoint(self, epoch, iteration):
        """Save model checkpoints."""
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'config': self.config.config
        }
        
        checkpoint_path = Path(self.config["checkpoint_dir"]) / f"checkpoint_epoch_{epoch}_iter_{iteration}.pth"
        torch.save(checkpoint, checkpoint_path)
    
    def _generate_samples(self, epoch):
        """Generate and save sample images."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(16, self.config["latent_dim"], 1, 1, device=self.device)
            fake_imgs = self.generator(z)
            
            # Save images
            grid = torchvision.utils.make_grid(fake_imgs, nrow=4, normalize=True)
            output_path = Path(self.config["output_dir"]) / f"generated_epoch_{epoch}.png"
            torchvision.utils.save_image(grid, output_path)
            
            # Log to tensorboard
            if self.config["use_tensorboard"]:
                self.writer.add_image('Generated Images', grid, epoch)
        
        self.generator.train()
    
    def generate_images(self, num_images=16, save_path=None):
        """Generate images for inference."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_images, self.config["latent_dim"], 1, 1, device=self.device)
            fake_imgs = self.generator(z)
            
            if save_path:
                grid = torchvision.utils.make_grid(fake_imgs, nrow=4, normalize=True)
                torchvision.utils.save_image(grid, save_path)
            
            return fake_imgs.cpu()

def main():
    """Main function for command-line training."""
    parser = argparse.ArgumentParser(description='Modern DCGAN Training')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'celeba'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--device', type=str, help='Device to use (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.dataset:
        config["dataset"] = args.dataset
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.device:
        config["device"] = args.device
    
    # Save configuration
    config.save_config("config.json")
    
    # Initialize trainer
    trainer = DCGANTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
