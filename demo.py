#!/usr/bin/env python3
"""
Demo Script for Modern DCGAN
============================

This script demonstrates the capabilities of the modern DCGAN implementation.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from modern_dcgan import Config, DCGANTrainer, ModernGenerator

def demo_quick_training():
    """Demonstrate quick training with minimal epochs."""
    print("🚀 Starting DCGAN Demo Training...")
    
    # Create configuration for quick demo
    config = Config()
    config["dataset"] = "mnist"  # Use MNIST for faster training
    config["num_epochs"] = 2     # Just 2 epochs for demo
    config["batch_size"] = 64
    config["use_tensorboard"] = False  # Disable for demo
    config["use_wandb"] = False
    config["log_interval"] = 50
    config["save_interval"] = 200
    
    # Initialize trainer
    trainer = DCGANTrainer(config)
    
    # Start training
    trainer.train()
    
    print("✅ Demo training completed!")
    return trainer

def demo_image_generation(trainer):
    """Demonstrate image generation."""
    print("🎨 Generating sample images...")
    
    # Generate images
    images = trainer.generate_images(num_images=16, save_path="demo_generated.png")
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    
    for i, img in enumerate(images):
        # Convert from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy
        if img.dim() == 3:
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img.squeeze().numpy()
        
        axes[i].imshow(img_np, cmap='gray' if img_np.shape[-1] == 1 else None)
        axes[i].axis('off')
    
    plt.suptitle("🎨 DCGAN Generated Images", fontsize=16)
    plt.tight_layout()
    plt.savefig("demo_grid.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Images generated and saved!")

def demo_model_info(trainer):
    """Display model information."""
    print("📊 Model Information:")
    print(f"  Generator parameters: {sum(p.numel() for p in trainer.generator.parameters()):,}")
    print(f"  Discriminator parameters: {sum(p.numel() for p in trainer.discriminator.parameters()):,}")
    print(f"  Total parameters: {sum(p.numel() for p in trainer.generator.parameters()) + sum(p.numel() for p in trainer.discriminator.parameters()):,}")
    print(f"  Device: {trainer.device}")
    print(f"  Dataset: {trainer.config['dataset']}")
    print(f"  Image size: {trainer.config['image_size']}x{trainer.config['image_size']}")

def demo_latent_space_exploration(trainer):
    """Demonstrate latent space interpolation."""
    print("🔍 Exploring latent space...")
    
    trainer.generator.eval()
    
    # Create two random latent vectors
    z1 = torch.randn(1, trainer.config["latent_dim"], 1, 1, device=trainer.device)
    z2 = torch.randn(1, trainer.config["latent_dim"], 1, 1, device=trainer.device)
    
    # Interpolate between them
    num_steps = 8
    interpolated_images = []
    
    with torch.no_grad():
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = trainer.generator(z_interp)
            interpolated_images.append(img.cpu())
    
    # Create interpolation visualization
    fig, axes = plt.subplots(1, num_steps, figsize=(12, 2))
    
    for i, img in enumerate(interpolated_images):
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        
        if img.dim() == 4:
            img = img.squeeze(0)
        
        img_np = img.permute(1, 2, 0).numpy() if img.dim() == 3 else img.squeeze().numpy()
        
        axes[i].imshow(img_np, cmap='gray' if img_np.shape[-1] == 1 else None)
        axes[i].axis('off')
        axes[i].set_title(f"α={i/(num_steps-1):.1f}")
    
    plt.suptitle("🔍 Latent Space Interpolation", fontsize=14)
    plt.tight_layout()
    plt.savefig("demo_interpolation.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Latent space exploration completed!")

def main():
    """Main demo function."""
    print("🎨 Modern DCGAN Demo")
    print("=" * 30)
    
    try:
        # Quick training demo
        trainer = demo_quick_training()
        
        # Model information
        demo_model_info(trainer)
        
        # Image generation demo
        demo_image_generation(trainer)
        
        # Latent space exploration
        demo_latent_space_exploration(trainer)
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run full training: python train.py --epochs 100")
        print("2. Launch interactive UI: streamlit run streamlit_app.py")
        print("3. View TensorBoard: tensorboard --logdir logs")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
