#!/usr/bin/env python3
"""
Setup Script for Modern DCGAN Project
====================================

This script sets up the project environment and creates necessary directories.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "outputs", 
        "checkpoints",
        "logs",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def create_default_config():
    """Create default configuration."""
    try:
        from modern_dcgan import Config
        config = Config()
        config.save_config("config.json")
        print("✅ Created default config.json")
        return True
    except ImportError:
        print("❌ Could not import modern_dcgan module")
        return False

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} is installed")
        
        # Check for CUDA
        if torch.cuda.is_available():
            print(f"🚀 CUDA {torch.version.cuda} is available")
        else:
            print("💻 CUDA not available, will use CPU")
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 MPS (Apple Silicon) is available")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def main():
    """Main setup function."""
    print("🎨 Modern DCGAN Project Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return 1
    
    # Check PyTorch
    print("\n🔍 Checking PyTorch installation...")
    if not check_pytorch():
        print("❌ PyTorch installation issues")
        return 1
    
    # Create default config
    print("\n⚙️ Creating default configuration...")
    if not create_default_config():
        print("❌ Failed to create configuration")
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run training: python train.py")
    print("2. Launch UI: streamlit run streamlit_app.py")
    print("3. View TensorBoard: tensorboard --logdir logs")
    
    return 0

if __name__ == "__main__":
    exit(main())
