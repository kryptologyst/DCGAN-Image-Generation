"""
Streamlit UI for Modern DCGAN Image Generation
============================================

Interactive web interface for generating images with the trained DCGAN model.
"""

import streamlit as st
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from pathlib import Path
import json

# Import our DCGAN classes
from modern_dcgan import Config, ModernGenerator, ModernDiscriminator

# Page configuration
st.set_page_config(
    page_title="🎨 Modern DCGAN Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .generated-image {
        border: 2px solid #1f77b4;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(config_path="config.json", checkpoint_path=None):
    """Load the trained DCGAN model."""
    try:
        # Load configuration
        config = Config(config_path)
        
        # Initialize models
        generator = ModernGenerator(config)
        discriminator = ModernDiscriminator(config)
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            st.success(f"✅ Model loaded from {checkpoint_path}")
        else:
            st.warning("⚠️ No checkpoint found. Using untrained model.")
        
        generator.eval()
        return generator, discriminator, config
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def generate_images(generator, config, num_images=16, seed=None):
    """Generate images using the DCGAN generator."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        z = torch.randn(num_images, config["latent_dim"], 1, 1)
        fake_imgs = generator(z)
        return fake_imgs

def create_image_grid(images, nrow=4):
    """Create a grid of images."""
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True, padding=2)
    return grid

def tensor_to_pil(tensor):
    """Convert PyTorch tensor to PIL Image."""
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    if tensor.dim() == 4:  # Batch of images
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and transpose for PIL
    np_image = tensor.permute(1, 2, 0).numpy()
    np_image = (np_image * 255).astype(np.uint8)
    
    return Image.fromarray(np_image)

def get_image_download_link(img_array, filename="generated_image.png"):
    """Create a download link for the image."""
    img_pil = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Image</a>'
    return href

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎨 Modern DCGAN Image Generator</h1>', unsafe_allow_html=True)
    st.markdown("Generate realistic images using Deep Convolutional Generative Adversarial Networks")
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Model selection
    st.sidebar.markdown("### 📁 Model Selection")
    config_path = st.sidebar.text_input("Config Path", value="config.json")
    
    # Check for available checkpoints
    checkpoint_dir = Path("checkpoints")
    available_checkpoints = []
    if checkpoint_dir.exists():
        available_checkpoints = list(checkpoint_dir.glob("*.pth"))
    
    if available_checkpoints:
        checkpoint_options = ["None (Untrained Model)"] + [str(cp) for cp in available_checkpoints]
        selected_checkpoint = st.sidebar.selectbox("Select Checkpoint", checkpoint_options)
        checkpoint_path = None if selected_checkpoint == "None (Untrained Model)" else selected_checkpoint
    else:
        st.sidebar.info("No checkpoints found. Using untrained model.")
        checkpoint_path = None
    
    # Load model
    generator, discriminator, config = load_model(config_path, checkpoint_path)
    
    if generator is None:
        st.error("Failed to load model. Please check your configuration.")
        return
    
    # Generation parameters
    st.sidebar.markdown("### 🎲 Generation Parameters")
    
    num_images = st.sidebar.slider("Number of Images", min_value=1, max_value=64, value=16, step=1)
    use_seed = st.sidebar.checkbox("Use Fixed Seed", value=False)
    seed = None
    if use_seed:
        seed = st.sidebar.number_input("Seed", min_value=0, max_value=2**32-1, value=42)
    
    # Advanced parameters
    with st.sidebar.expander("🔧 Advanced Parameters"):
        latent_dim = st.number_input("Latent Dimension", min_value=10, max_value=512, value=config["latent_dim"])
        temperature = st.slider("Temperature (Noise Scale)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎨 Generated Images</h2>', unsafe_allow_html=True)
        
        # Generate button
        if st.button("🚀 Generate Images", type="primary"):
            with st.spinner("Generating images..."):
                # Generate images
                generated_images = generate_images(generator, config, num_images, seed)
                
                # Create grid
                grid = create_image_grid(generated_images, nrow=4)
                
                # Convert to PIL and display
                pil_image = tensor_to_pil(grid)
                
                st.image(pil_image, caption="Generated Images", use_column_width=True)
                
                # Download button
                img_array = np.array(pil_image)
                download_link = get_image_download_link(img_array, f"dcgan_generated_{num_images}images.png")
                st.markdown(download_link, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Model Info</h2>', unsafe_allow_html=True)
        
        # Model statistics
        st.markdown("### Model Statistics")
        st.metric("Latent Dimension", config["latent_dim"])
        st.metric("Image Size", f"{config['image_size']}x{config['image_size']}")
        st.metric("Channels", config["channels"])
        st.metric("Generator Features", config["generator_features"])
        st.metric("Discriminator Features", config["discriminator_features"])
        
        # Training info
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            st.markdown("### Training Info")
            st.metric("Epoch", checkpoint.get('epoch', 'Unknown'))
            st.metric("Iteration", checkpoint.get('iteration', 'Unknown'))
        
        # Model architecture
        st.markdown("### Architecture")
        st.code(f"""
Generator: {sum(p.numel() for p in generator.parameters()):,} parameters
Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} parameters
Total: {sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters()):,} parameters
        """)
    
    # Additional features
    st.markdown("---")
    
    # Batch generation
    st.markdown('<h2 class="sub-header">🔄 Batch Generation</h2>', unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("Generate 4 Images"):
            with st.spinner("Generating..."):
                images = generate_images(generator, config, 4, seed)
                grid = create_image_grid(images, nrow=2)
                pil_image = tensor_to_pil(grid)
                st.image(pil_image, caption="4 Generated Images")
    
    with col4:
        if st.button("Generate 9 Images"):
            with st.spinner("Generating..."):
                images = generate_images(generator, config, 9, seed)
                grid = create_image_grid(images, nrow=3)
                pil_image = tensor_to_pil(grid)
                st.image(pil_image, caption="9 Generated Images")
    
    with col5:
        if st.button("Generate 16 Images"):
            with st.spinner("Generating..."):
                images = generate_images(generator, config, 16, seed)
                grid = create_image_grid(images, nrow=4)
                pil_image = tensor_to_pil(grid)
                st.image(pil_image, caption="16 Generated Images")
    
    # Interactive exploration
    st.markdown('<h2 class="sub-header">🔍 Interactive Exploration</h2>', unsafe_allow_html=True)
    
    st.markdown("Explore the latent space by adjusting individual dimensions:")
    
    # Create sliders for latent dimensions
    if st.checkbox("Show Latent Space Controls"):
        cols = st.columns(4)
        latent_values = []
        
        for i in range(min(8, config["latent_dim"])):  # Show first 8 dimensions
            with cols[i % 4]:
                value = st.slider(f"Dim {i}", -3.0, 3.0, 0.0, 0.1, key=f"latent_{i}")
                latent_values.append(value)
        
        # Pad with random values for remaining dimensions
        while len(latent_values) < config["latent_dim"]:
            latent_values.append(0.0)
        
        if st.button("Generate with Custom Latent"):
            with st.spinner("Generating with custom latent vector..."):
                # Create custom latent vector
                z_custom = torch.tensor(latent_values).reshape(1, -1, 1, 1)
                
                with torch.no_grad():
                    custom_image = generator(z_custom)
                    pil_image = tensor_to_pil(custom_image)
                    st.image(pil_image, caption="Custom Latent Generation")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎨 Modern DCGAN Image Generator | Built with PyTorch & Streamlit</p>
        <p>Generate realistic images using state-of-the-art GAN techniques</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
