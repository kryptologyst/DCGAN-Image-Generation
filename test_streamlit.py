"""
Simplified Streamlit App for DCGAN Testing
==========================================

This is a minimal version to test if Streamlit works with our DCGAN code.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="🎨 DCGAN Test",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 DCGAN Image Generator - Test Version")

# Simple test without loading the full model
st.write("This is a simplified test version of the DCGAN app.")

# Test basic functionality
if st.button("Test Basic Functionality"):
    st.success("✅ Streamlit is working!")
    
    # Test PyTorch
    try:
        x = torch.randn(1, 3, 64, 64)
        st.success(f"✅ PyTorch is working! Generated tensor shape: {x.shape}")
    except Exception as e:
        st.error(f"❌ PyTorch error: {str(e)}")
    
    # Test matplotlib
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")
        st.pyplot(fig)
        st.success("✅ Matplotlib is working!")
    except Exception as e:
        st.error(f"❌ Matplotlib error: {str(e)}")

# Test model loading
if st.button("Test Model Loading"):
    try:
        # Try to import our modules
        from modern_dcgan import Config, ModernGenerator
        st.success("✅ DCGAN modules imported successfully!")
        
        # Try to create a simple config
        config = Config()
        st.success(f"✅ Config created! Dataset: {config['dataset']}")
        
        # Try to create a generator (without loading weights)
        generator = ModernGenerator(config)
        st.success("✅ Generator created successfully!")
        
        # Test generation
        with torch.no_grad():
            z = torch.randn(1, config["latent_dim"], 1, 1)
            fake_img = generator(z)
            st.success(f"✅ Image generation test successful! Output shape: {fake_img.shape}")
            
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")

st.write("---")
st.write("If all tests pass, the full Streamlit app should work. If not, we need to fix the underlying issues.")
