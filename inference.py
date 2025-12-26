import torch
import os
# Force the system to use the first available GPU (device 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import argparse
import numpy as np
from PIL import Image
from diffusers import (
    ControlNetModel, 
    StableDiffusionXLControlNetPipeline,
    DDIMScheduler
)
from ip_adapter import IPAdapterXL
from utils import resize_image, empty_cache

# --- Setup Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--target_blocks", nargs="+", type=str, default=["up_blocks.0.attentions.1"]) 
parser.add_argument("--scale", type=float, default=1.2) 
parser.add_argument("--style_image_id", type=int, default=103)
parser.add_argument("--contet_image_id", type=int, default=14)
args = parser.parse_args()

# --- Paths ---
sdxl_base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_path ="xinsir/controlnet-tile-sdxl-1.0"
ip_adapter_extractor_path = "IP-Adapter/sdxl_models/image_encoder"
ip_adapter_module_path = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

# --- Device Handling ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Models ---
# 1. Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    controlnet_path, 
    torch_dtype=torch.float16
).to(device)

# 2. Load Pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    sdxl_base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

# --- Performance Optimizations ---
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Updated for modern diffusers versions
pipe.vae.enable_tiling() 

if device == "cuda":
    # FIXED: Replaced xformers with native attention slicing to prevent ModuleNotFoundError
    pipe.enable_attention_slicing()

# 3. Load IP-Adapter
model = IPAdapterXL(
    pipe,
    ip_adapter_extractor_path,
    ip_adapter_module_path,
    device,
    target_blocks=args.target_blocks
)

# Free up CPU RAM after moving models to GPU
empty_cache()

# --- Image Processing ---
style_image_path = f'data/style/{args.style_image_id}.jpg'
content_image_path = f'data/content/{args.contet_image_id}.jpg'

style_image = Image.open(style_image_path).convert("RGB")
content_image = Image.open(content_image_path).convert("RGB")
H, W = content_image.size

# Resize for SDXL
controlnet_cond_image = resize_image(content_image, short=1024)

input_kwargs = {
    'pil_image': style_image,
    'image': controlnet_cond_image,
}

# --- Generation ---
print("Starting generation on GPU...")
with torch.no_grad():
    generated = model.generate(
        prompt="masterpiece, best quality, van gogh style, oil painting",
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, blurry",
        scale=args.scale,
        guidance_scale=5.0,
        num_samples=1,
        num_inference_steps=30, 
        seed=42,
        controlnet_conditioning_scale=0.6,
        **input_kwargs
    )

# --- Save Results ---
os.makedirs("results", exist_ok=True)
generated_img = generated[0].resize((H, W), resample=Image.Resampling.LANCZOS)
output_filename = f"results/output_{args.style_image_id}_{args.contet_image_id}_{args.scale}.png"
generated_img.save(output_filename)
print(f"Success! Image saved to {output_filename}")