#!/usr/bin/env python3
"""
Simple DCGAN Test Script
========================

Test the core DCGAN functionality without display components.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_basic_functionality():
    """Test basic PyTorch functionality."""
    print("🔍 Testing basic PyTorch functionality...")
    
    try:
        # Test tensor creation
        x = torch.randn(2, 3, 64, 64)
        print(f"✅ Tensor creation successful: {x.shape}")
        
        # Test device detection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Device detection successful: {device}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_dcgan_modules():
    """Test DCGAN module imports and creation."""
    print("🔍 Testing DCGAN modules...")
    
    try:
        from modern_dcgan import Config, ModernGenerator, ModernDiscriminator
        
        # Create configuration
        config = Config()
        print(f"✅ Config created: dataset={config['dataset']}")
        
        # Create models
        generator = ModernGenerator(config)
        discriminator = ModernDiscriminator(config)
        print(f"✅ Models created successfully")
        
        # Test forward pass
        z = torch.randn(1, config["latent_dim"], 1, 1)
        fake_img = generator(z)
        print(f"✅ Generator forward pass successful: {fake_img.shape}")
        
        real_img = torch.randn(1, config["channels"], config["image_size"], config["image_size"])
        disc_out = discriminator(real_img)
        print(f"✅ Discriminator forward pass successful: {disc_out.shape}")
        
        return generator, discriminator, config
        
    except Exception as e:
        print(f"❌ DCGAN module test failed: {e}")
        return None, None, None

def test_image_generation(generator, config):
    """Test image generation without display."""
    print("🔍 Testing image generation...")
    
    try:
        generator.eval()
        with torch.no_grad():
            z = torch.randn(16, config["latent_dim"], 1, 1)
            fake_imgs = generator(z)
            
            # Convert to numpy for analysis
            fake_imgs_np = fake_imgs.cpu().numpy()
            
            print(f"✅ Generated {fake_imgs_np.shape[0]} images")
            print(f"✅ Image shape: {fake_imgs_np.shape[1:]} (channels, height, width)")
            print(f"✅ Value range: [{fake_imgs_np.min():.3f}, {fake_imgs_np.max():.3f}]")
            
            # Save as numpy array
            np.save("generated_images.npy", fake_imgs_np)
            print("✅ Images saved as generated_images.npy")
            
            return True
            
    except Exception as e:
        print(f"❌ Image generation test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🎨 DCGAN Core Functionality Test")
    print("=" * 40)
    
    # Test 1: Basic functionality
    if not test_basic_functionality():
        print("❌ Basic functionality test failed. Exiting.")
        return 1
    
    # Test 2: DCGAN modules
    generator, discriminator, config = test_dcgan_modules()
    if generator is None:
        print("❌ DCGAN module test failed. Exiting.")
        return 1
    
    # Test 3: Image generation
    if not test_image_generation(generator, config):
        print("❌ Image generation test failed.")
        return 1
    
    print("\n🎉 All tests passed!")
    print("\n📊 Model Statistics:")
    print(f"  Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"  Total parameters: {sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"  Latent dimension: {config['latent_dim']}")
    print(f"  Image size: {config['image_size']}x{config['image_size']}")
    print(f"  Channels: {config['channels']}")
    
    print("\n✅ DCGAN is working correctly!")
    print("📁 Generated images saved as 'generated_images.npy'")
    
    return 0

if __name__ == "__main__":
    exit(main())
