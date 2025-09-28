import os
import subprocess
import tempfile
from typing import Optional
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from cog import BasePredictor, Input, Path as CogPath


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Model configuration - use pre-downloaded weights from build time
        self.model_id = "/app/model-weights"
        
        # Verify model weights exist (should be downloaded during build)
        if not os.path.exists(self.model_id):
            raise RuntimeError(
                "Model weights not found at /app/model-weights. "
                "They should have been downloaded during the build process. "
                "Please check the cog.yaml run command."
            )
        
        print("Loading HunyuanImage-3.0 model...")
        
        # Model loading configuration
        kwargs = {
            "attn_implementation": "flash_attention_2",  # Use FlashAttention if available
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "device_map": "auto",
            "moe_impl": "flashinfer",  # Use FlashInfer for optimized inference
        }
        
        try:
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            self.model.load_tokenizer(self.model_id)
            print("Model loaded successfully with FlashAttention and FlashInfer optimizations")
        except Exception as e:
            print(f"Failed to load with optimizations: {e}")
            # Fallback to basic configuration
            kwargs.update({
                "attn_implementation": "sdpa",
                "moe_impl": "eager"
            })
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            self.model.load_tokenizer(self.model_id)
            print("Model loaded successfully with basic configuration")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for image generation",
            default="A beautiful landscape with mountains and a lake"
        ),
        image_size: str = Input(
            description="Image size (e.g., '1280x768', '1024x1024', or 'auto')",
            default="auto",
            choices=["auto", "1024x1024", "1280x768", "768x1280", "1536x640", "640x1536"]
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=50,
            ge=1,
            le=200
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible generation",
            default=None
        ),
        verbose: bool = Input(
            description="Enable verbose output",
            default=False
        )
    ) -> CogPath:
        """Run a single prediction on the model"""
        
        if verbose:
            print(f"Generating image with prompt: '{prompt}'")
            print(f"Image size: {image_size}")
            print(f"Inference steps: {num_inference_steps}")
            print(f"Seed: {seed}")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Generate image
        try:
            # Create generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "stream": True
            }
            
            # Add optional parameters if they differ from defaults
            if image_size != "auto":
                generation_kwargs["image_size"] = image_size
            if num_inference_steps != 50:
                generation_kwargs["diff_infer_steps"] = num_inference_steps
            
            # Generate the image
            if verbose:
                print("Starting image generation...")
            
            image = self.model.generate_image(**generation_kwargs)
            
            if verbose:
                print("Image generation completed")
            
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                output_path = tmp_file.name
                image.save(output_path, format="PNG")
            
            if verbose:
                print(f"Image saved to: {output_path}")
            
            return CogPath(output_path)
            
        except Exception as e:
            print(f"Error during image generation: {e}")
            raise e
