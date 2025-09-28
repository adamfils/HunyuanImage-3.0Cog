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
        
        # Check CUDA availability and set device mapping accordingly
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"CUDA available with {device_count} GPU(s)")
            
            # Explicit device mapping for multi-GPU setups
            if device_count > 1:
                device_map = {}
                # Distribute model layers across available GPUs
                for i in range(device_count):
                    device_map[f"model.layers.{i}"] = i % device_count
                device_map["model.embed_tokens"] = 0
                device_map["model.norm"] = device_count - 1
                device_map["lm_head"] = device_count - 1
            else:
                device_map = {"": 0}  # Single GPU mapping
        else:
            print("CUDA not available, using CPU")
            device_map = {"": "cpu"}
        
        # Model loading configuration
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        
        # Add optimizations if CUDA is available
        if torch.cuda.is_available():
            kwargs.update({
                "attn_implementation": "flash_attention_2",  # Use FlashAttention if available
                "moe_impl": "flashinfer",  # Use FlashInfer for optimized inference
            })
        
        try:
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            self.model.load_tokenizer(self.model_id)
            print("Model loaded successfully with optimizations" if torch.cuda.is_available() else "Model loaded successfully on CPU")
        except Exception as e:
            print(f"Failed to load with optimizations: {e}")
            # Fallback to basic configuration
            kwargs.update({
                "attn_implementation": "eager" if torch.cuda.is_available() else "eager",
                "moe_impl": "eager"
            })
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
                self.model.load_tokenizer(self.model_id)
                print("Model loaded successfully with fallback configuration")
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                # Final fallback - CPU only
                kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float32,
                    "device_map": {"": "cpu"},
                    "low_cpu_mem_usage": True,
                }
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
                self.model.load_tokenizer(self.model_id)
                print("Model loaded successfully on CPU (final fallback)")

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
