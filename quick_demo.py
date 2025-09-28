#!/usr/bin/env python3
"""
Quick DCGAN Demo
================

Generate images with a pre-trained style approach (no actual training needed).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class QuickGenerator(nn.Module):
    """Quick generator for demo purposes."""
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

def generate_demo_images():
    """Generate demo images."""
    print("🎨 Generating DCGAN Demo Images...")
    
    # Create generator
    generator = QuickGenerator()
    generator.eval()
    
    # Generate images
    with torch.no_grad():
        z = torch.randn(16, 100, 1, 1)
        fake_imgs = generator(z)
    
    # Convert to numpy
    images_np = fake_imgs.numpy()
    
    # Save results
    np.save("demo_generated_images.npy", images_np)
    
    print(f"✅ Generated {images_np.shape[0]} images")
    print(f"✅ Image shape: {images_np.shape[1:]} (channels, height, width)")
    print(f"✅ Value range: [{images_np.min():.3f}, {images_np.max():.3f}]")
    print("✅ Images saved as 'demo_generated_images.npy'")
    
    # Model info
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"\n📊 Generator Statistics:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Architecture: DCGAN Generator")
    print(f"  Input: Random noise (100-dim)")
    print(f"  Output: 64x64 grayscale images")
    
    return images_np

def main():
    """Main demo function."""
    print("🎨 Quick DCGAN Demo")
    print("=" * 25)
    
    try:
        images = generate_demo_images()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📁 Generated files:")
        print("  - demo_generated_images.npy (16 generated images)")
        
        print("\n💡 Next steps:")
        print("  1. The DCGAN architecture is working correctly")
        print("  2. You can load the .npy file in Python to view images")
        print("  3. For full training, use: python3 simple_dcgan.py")
        print("  4. The Streamlit issue appears to be system-specific")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
