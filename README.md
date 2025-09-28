# HunyuanImage-3.0 Cog Implementation

A [Cog](https://github.com/replicate/cog) implementation of [Tencent's HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0) text-to-image generation model for deployment on [Replicate](https://replicate.com).

## Model Overview

HunyuanImage-3.0 is a powerful native multimodal model for image generation developed by Tencent. This implementation wraps the model in a Cog container for easy deployment and serving.

### Key Features
- 🎨 High-quality text-to-image generation
- 🚀 Optimized with FlashAttention and FlashInfer for 3x faster inference
- 📐 Multiple image size options (1024x1024, 1280x768, custom ratios)
- 🎯 Configurable inference steps and random seed support
- 🔧 Automatic model downloading and setup

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **GPU Memory**: ≥3×80GB (4×80GB recommended for optimal performance)
- **Disk Space**: ~170GB for model weights
- **CUDA**: 12.8
- **Python**: 3.12+

## Quick Start

### Prerequisites

1. Install [Docker](https://docs.docker.com/get-docker/) with GPU support
2. Install [Cog](https://github.com/replicate/cog):
   ```bash
   # macOS
   brew install cog
   
   # Linux/other
   sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
   sudo chmod +x /usr/local/bin/cog
   ```

### Local Testing

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd HunyuanImage-3.0Cog
   ```

2. Run a prediction:
   ```bash
   cog predict -i prompt="A beautiful landscape with mountains and a lake"
   ```

3. Build the Docker image:
   ```bash
   cog build -t hunyuan-image-3
   ```

4. Run the container locally:
   ```bash
   cog run -p 5000 hunyuan-image-3
   ```

### API Usage

Once running, you can make HTTP requests to generate images:

```bash
curl http://localhost:5000/predictions -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "prompt": "A futuristic cityscape at sunset",
      "image_size": "1280x768",
      "num_inference_steps": 50,
      "seed": 42
    }
  }'
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | `"A beautiful landscape with mountains and a lake"` | Text prompt for image generation |
| `image_size` | `str` | `"auto"` | Image dimensions (`auto`, `1024x1024`, `1280x768`, `768x1280`, `1536x640`, `640x1536`) |
| `num_inference_steps` | `int` | `50` | Number of denoising steps (1-200) |
| `seed` | `int` | `None` | Random seed for reproducible generation |
| `verbose` | `bool` | `False` | Enable detailed logging |

## Deployment on Replicate

### 1. Push to GitHub

Ensure your code is in a public GitHub repository.

### 2. Create Model on Replicate

1. Go to [replicate.com](https://replicate.com)
2. Click "Create a model"
3. Connect your GitHub repository
4. Replicate will automatically build and deploy your model

### 3. Alternative: Manual Deployment

```bash
# Build and push to a registry
cog build -t your-registry/hunyuan-image-3

# Deploy using Replicate's API
replicate models create your-username/hunyuan-image-3
replicate models versions create your-username/hunyuan-image-3 --image your-registry/hunyuan-image-3
```

## Performance Optimizations

This implementation includes several performance optimizations:

- **FlashAttention 2.8.3**: Faster attention computation
- **FlashInfer**: Optimized mixture-of-experts inference
- **Automatic fallback**: Falls back to basic configuration if optimizations fail
- **GPU memory management**: Automatic device mapping for multi-GPU setups

### First Run Notice

⚠️ **Important**: When FlashInfer is enabled, the first inference may take ~10 minutes due to kernel compilation. Subsequent inferences will be much faster.

## File Structure

```
HunyuanImage-3.0Cog/
├── cog.yaml          # Cog configuration with dependencies
├── predict.py        # Main predictor class
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Development

### Local Development Setup

1. Create a virtual environment:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Test the predictor locally:
   ```python
   from predict import Predictor
   
   predictor = Predictor()
   predictor.setup()
   
   result = predictor.predict(
       prompt="A serene forest scene",
       image_size="1024x1024"
   )
   print(f"Generated image: {result}")
   ```

### Model Download

The model weights (~170GB) are automatically downloaded from Hugging Face on first run. The download includes:
- Model weights and configuration
- Tokenizer files
- Additional model components

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use smaller image dimensions
   - Ensure you have sufficient GPU memory (≥240GB total recommended)

2. **Model Download Fails**
   - Check internet connection
   - Verify Hugging Face Hub access
   - Ensure sufficient disk space (170GB+)

3. **FlashAttention Installation Issues**
   - Ensure CUDA toolkit is properly installed
   - Check GCC version (≥9 recommended)
   - Verify PyTorch CUDA version matches system CUDA

4. **First Inference Very Slow**
   - This is expected with FlashInfer optimization
   - Subsequent inferences will be significantly faster

### Getting Help

- Check [Cog documentation](https://github.com/replicate/cog)
- Review [HunyuanImage-3.0 model page](https://huggingface.co/tencent/HunyuanImage-3.0)
- Open an issue in this repository

## License

This implementation follows the licensing of the original HunyuanImage-3.0 model. Please refer to the [original model repository](https://huggingface.co/tencent/HunyuanImage-3.0) for license details.

## Acknowledgments

- [Tencent](https://github.com/Tencent-Hunyuan) for the HunyuanImage-3.0 model
- [Replicate](https://replicate.com) for the Cog framework
- The open-source community for FlashAttention and FlashInfer optimizations
