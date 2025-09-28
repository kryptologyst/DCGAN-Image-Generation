#!/usr/bin/env python3
"""
Training Script for Modern DCGAN
===============================

This script provides an easy way to train the DCGAN model with different configurations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from modern_dcgan import Config, DCGANTrainer

def create_default_config():
    """Create a default configuration file."""
    config = Config()
    config.save_config("config.json")
    print("✅ Created default config.json")

def main():
    parser = argparse.ArgumentParser(description="Train Modern DCGAN")
    
    # Configuration options
    parser.add_argument("--config", type=str, default="config.json", 
                       help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true",
                       help="Create default configuration file and exit")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10", "celeba"],
                       help="Dataset to use for training")
    
    # Training options
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for training")
    
    # Model options
    parser.add_argument("--latent-dim", type=int, help="Latent dimension")
    parser.add_argument("--image-size", type=int, help="Image size")
    parser.add_argument("--channels", type=int, help="Number of channels")
    
    # Advanced options
    parser.add_argument("--use-spectral-norm", action="store_true",
                       help="Use spectral normalization")
    parser.add_argument("--use-gradient-penalty", action="store_true",
                       help="Use gradient penalty")
    parser.add_argument("--n-critic", type=int, help="Number of discriminator updates per generator update")
    
    # Monitoring options
    parser.add_argument("--use-tensorboard", action="store_true",
                       help="Enable TensorBoard logging")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--log-interval", type=int, help="Logging interval")
    parser.add_argument("--save-interval", type=int, help="Checkpoint saving interval")
    
    # Paths
    parser.add_argument("--data-path", type=str, help="Path to dataset")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Create config if requested
    if args.create_config:
        create_default_config()
        return
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"❌ Configuration file {args.config} not found!")
        print("Use --create-config to create a default configuration file.")
        return
    
    config = Config(args.config)
    
    # Override config with command line arguments
    overrides = {
        "dataset": args.dataset,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": args.device,
        "latent_dim": args.latent_dim,
        "image_size": args.image_size,
        "channels": args.channels,
        "n_critic": args.n_critic,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "checkpoint_dir": args.checkpoint_dir,
    }
    
    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    
    # Set boolean flags
    if args.use_spectral_norm:
        config["use_spectral_norm"] = True
    if args.use_gradient_penalty:
        config["use_gradient_penalty"] = True
    if args.use_tensorboard:
        config["use_tensorboard"] = True
    if args.use_wandb:
        config["use_wandb"] = True
    
    # Save updated config
    config.save_config(args.config)
    
    # Print configuration
    print("🔧 Training Configuration:")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Device: {config['device']}")
    print(f"  Latent Dim: {config['latent_dim']}")
    print(f"  Image Size: {config['image_size']}")
    print(f"  Channels: {config['channels']}")
    print(f"  Spectral Norm: {config['use_spectral_norm']}")
    print(f"  Gradient Penalty: {config['use_gradient_penalty']}")
    print(f"  TensorBoard: {config['use_tensorboard']}")
    print(f"  Weights & Biases: {config['use_wandb']}")
    print()
    
    # Initialize trainer
    try:
        trainer = DCGANTrainer(config)
        print("✅ Trainer initialized successfully!")
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
