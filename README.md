# DCGAN Image Generation

A state-of-the-art implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) with modern PyTorch features, advanced training techniques, and an interactive web interface.

## Features

- **Modern PyTorch 2.x** with `torch.compile` for faster training
- **Spectral Normalization** for improved training stability
- **Gradient Penalty** for better convergence
- **Multiple Dataset Support** (MNIST, CIFAR-10, CelebA)
- **Interactive Streamlit UI** for real-time image generation
- **TensorBoard Integration** for training monitoring
- **Weights & Biases** support for experiment tracking
- **Comprehensive Configuration System** for easy customization
- **Automatic Checkpointing** and model saving
- **Cross-platform Support** (CPU, CUDA, MPS)

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd 0133_DCGAN_for_image_generation
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create default configuration:**
```bash
python train.py --create-config
```

### Training

**Basic training with default settings:**
```bash
python train.py
```

**Training with custom parameters:**
```bash
python train.py --dataset cifar10 --epochs 100 --batch-size 64 --use-tensorboard
```

**Training on specific device:**
```bash
python train.py --device cuda --epochs 200
```

### Interactive UI

**Launch the Streamlit interface:**
```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501` to start generating images interactively!

## 📁 Project Structure

```
0133_DCGAN_for_image_generation/
├── modern_dcgan.py          # Main DCGAN implementation
├── streamlit_app.py         # Interactive web UI
├── train.py                 # Training script
├── 0133.py                  # Original implementation
├── requirements.txt         # Python dependencies
├── config.json              # Configuration file (created by train.py)
├── data/                    # Dataset storage
├── outputs/                 # Generated images
├── checkpoints/             # Model checkpoints
└── logs/                    # Training logs
```

## 🔧 Configuration

The project uses a comprehensive configuration system. Key parameters:

### Model Parameters
- `latent_dim`: Dimension of the latent space (default: 100)
- `image_size`: Size of generated images (default: 64)
- `channels`: Number of image channels (default: 3 for RGB)
- `generator_features`: Number of generator features (default: 64)
- `discriminator_features`: Number of discriminator features (default: 64)

### Training Parameters
- `batch_size`: Training batch size (default: 64)
- `num_epochs`: Number of training epochs (default: 200)
- `learning_rate`: Learning rate (default: 0.0002)
- `beta1`, `beta2`: Adam optimizer parameters (default: 0.5, 0.999)

### Advanced Features
- `use_spectral_norm`: Enable spectral normalization (default: True)
- `use_gradient_penalty`: Enable gradient penalty (default: True)
- `gradient_penalty_weight`: Weight for gradient penalty (default: 10.0)
- `n_critic`: Discriminator updates per generator update (default: 5)

### Dataset Options
- `dataset`: Dataset to use (`mnist`, `cifar10`, `celeba`)
- `data_path`: Path to store datasets (default: `./data`)

## Usage Examples

### Command Line Training

**MNIST digits generation:**
```bash
python train.py --dataset mnist --epochs 50 --batch-size 128
```

**CIFAR-10 natural images:**
```bash
python train.py --dataset cifar10 --epochs 200 --use-tensorboard --use-wandb
```

**CelebA face generation:**
```bash
python train.py --dataset celeba --epochs 300 --batch-size 32 --image-size 128
```

### Programmatic Usage

```python
from modern_dcgan import Config, DCGANTrainer

# Create configuration
config = Config()
config["dataset"] = "cifar10"
config["num_epochs"] = 100
config["use_tensorboard"] = True

# Initialize trainer
trainer = DCGANTrainer(config)

# Start training
trainer.train()

# Generate images
images = trainer.generate_images(num_images=16, save_path="generated.png")
```

## Web Interface

The Streamlit interface provides:

- **Real-time Image Generation**: Generate images with custom parameters
- **Model Information**: View model statistics and architecture details
- **Interactive Controls**: Adjust latent space dimensions manually
- **Batch Generation**: Generate multiple images at once
- **Download Options**: Save generated images locally
- **Checkpoint Management**: Load different trained models

### UI Features

1. **Configuration Panel**: Adjust generation parameters
2. **Model Selection**: Choose from available checkpoints
3. **Live Generation**: Real-time image generation with progress indicators
4. **Interactive Exploration**: Manual latent space manipulation
5. **Batch Operations**: Generate multiple image sets
6. **Download Support**: Save generated images in various formats

## Monitoring and Logging

### TensorBoard Integration

```bash
# Start TensorBoard
tensorboard --logdir logs

# View training metrics at http://localhost:6006
```

### Weights & Biases Integration

```bash
# Enable W&B logging
python train.py --use-wandb

# View experiments at https://wandb.ai
```

## Advanced Features

### Spectral Normalization

Improves training stability by constraining the Lipschitz constant of the discriminator:

```python
config["use_spectral_norm"] = True
```

### Gradient Penalty

Prevents mode collapse and improves convergence:

```python
config["use_gradient_penalty"] = True
config["gradient_penalty_weight"] = 10.0
```

### Progressive Training

Train with different image sizes for better quality:

```python
# Start with smaller images
config["image_size"] = 32
# Train for some epochs, then increase
config["image_size"] = 64
```

## Generated Examples

The model can generate various types of images:

- **MNIST**: Handwritten digits (28x28 grayscale)
- **CIFAR-10**: Natural images (32x32 RGB)
- **CelebA**: Human faces (64x64 RGB)

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   python train.py --batch-size 32 --device cpu
   ```

2. **Dataset Download Issues**:
   ```bash
   # Manual dataset download
   python -c "import torchvision; torchvision.datasets.CIFAR10(root='./data', download=True)"
   ```

3. **Model Loading Errors**:
   - Ensure checkpoint files are compatible
   - Check configuration parameters match training config

### Performance Tips

1. **Use GPU acceleration** when available
2. **Enable torch.compile** for faster training (PyTorch 2.0+)
3. **Adjust batch size** based on available memory
4. **Use mixed precision** for large models

## Results and Metrics

The model tracks several important metrics:

- **Generator Loss**: Measures how well the generator fools the discriminator
- **Discriminator Loss**: Measures discriminator's ability to distinguish real/fake
- **FID Score**: Fréchet Inception Distance for quality assessment
- **Training Time**: Total training duration
- **Memory Usage**: GPU/CPU memory consumption

## Contributing

Contributions are welcome! Please feel free to submit:

- Bug fixes
- Feature enhancements
- Documentation improvements
- New dataset support
- Performance optimizations

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Original DCGAN paper: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
- PyTorch team for the excellent deep learning framework
- Streamlit team for the amazing web app framework
- The open-source community for various contributions

## References

1. Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.
2. Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral normalization for generative adversarial networks. arXiv preprint arXiv:1802.05957.
3. Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017). Improved training of wasserstein gans. Advances in neural information processing systems, 30.


# DCGAN-Image-Generation
