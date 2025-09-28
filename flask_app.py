"""
Flask Web App for DCGAN Image Generation
======================================

A simple web interface using Flask instead of Streamlit to avoid the mutex lock issue.
"""

from flask import Flask, render_template_string, request, send_file, jsonify
import torch
import torch.nn as nn
import numpy as np
import io
import base64
from PIL import Image
import json

app = Flask(__name__)

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

# Initialize generator
generator = QuickGenerator()
generator.eval()

def generate_images(num_images=16, seed=None):
    """Generate images using the DCGAN generator."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        z = torch.randn(num_images, 100, 1, 1)
        fake_imgs = generator(z)
        
        # Convert to PIL images
        images = []
        for i in range(num_images):
            img = fake_imgs[i]
            # Convert from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            
            # Convert to numpy
            img_np = img.squeeze().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # Convert to PIL
            pil_img = Image.fromarray(img_np, mode='L')
            images.append(pil_img)
        
        return images

def image_to_base64(img):
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🎨 DCGAN Image Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 30px;
        }
        .controls {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 20px;
        }
        .image-container {
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        button {
            background-color: #1f77b4;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        button:hover {
            background-color: #0d5a8a;
        }
        input, select {
            padding: 8px;
            margin: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        .stats {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 DCGAN Image Generator</h1>
        <p>Generate realistic images using Deep Convolutional Generative Adversarial Networks</p>
    </div>
    
    <div class="controls">
        <h3>⚙️ Generation Controls</h3>
        <form id="generateForm">
            <label>Number of Images:</label>
            <select name="num_images">
                <option value="4">4 Images</option>
                <option value="9">9 Images</option>
                <option value="16" selected>16 Images</option>
                <option value="25">25 Images</option>
            </select>
            
            <label>Seed (optional):</label>
            <input type="number" name="seed" placeholder="Leave empty for random">
            
            <br><br>
            <button type="submit">🚀 Generate Images</button>
        </form>
    </div>
    
    <div id="results" style="display: none;">
        <h3>🎨 Generated Images</h3>
        <div id="imageGrid" class="image-grid"></div>
    </div>
    
    <div class="stats">
        <h3>📊 Model Information</h3>
        <p><strong>Architecture:</strong> DCGAN Generator</p>
        <p><strong>Parameters:</strong> 1,066,880</p>
        <p><strong>Input:</strong> Random noise (100-dimensional)</p>
        <p><strong>Output:</strong> 64×64 grayscale images</p>
        <p><strong>Framework:</strong> PyTorch</p>
    </div>

    <script>
        document.getElementById('generateForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const params = new URLSearchParams();
            params.append('num_images', formData.get('num_images'));
            if (formData.get('seed')) {
                params.append('seed', formData.get('seed'));
            }
            
            try {
                const response = await fetch('/generate?' + params.toString());
                const data = await response.json();
                
                if (data.success) {
                    const imageGrid = document.getElementById('imageGrid');
                    imageGrid.innerHTML = '';
                    
                    data.images.forEach((imgData, index) => {
                        const container = document.createElement('div');
                        container.className = 'image-container';
                        
                        const img = document.createElement('img');
                        img.src = imgData;
                        img.alt = `Generated Image ${index + 1}`;
                        
                        container.appendChild(img);
                        imageGrid.appendChild(container);
                    });
                    
                    document.getElementById('results').style.display = 'block';
                } else {
                    alert('Error generating images: ' + data.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate')
def generate():
    try:
        num_images = int(request.args.get('num_images', 16))
        seed = request.args.get('seed')
        seed = int(seed) if seed else None
        
        # Limit number of images for performance
        num_images = min(num_images, 25)
        
        # Generate images
        images = generate_images(num_images, seed)
        
        # Convert to base64
        image_data = [image_to_base64(img) for img in images]
        
        return jsonify({
            'success': True,
            'images': image_data,
            'count': len(images)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'DCGAN Flask app is running!'})

if __name__ == '__main__':
    print("🎨 Starting DCGAN Flask Web App...")
    print("📱 Open your browser to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
