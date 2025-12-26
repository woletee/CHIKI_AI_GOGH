import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import numpy as np
from PIL import Image
from PIL import Image
from diffusers import(
    ControlNetModel, 
    StableDiffusionXLControlNetPipeline,
    DDIMScheduler
)
from ip_adapter import IPAdapterXL
from utils import resize_image, empty_cache

parser = argparse.ArgumentParser()
parser.add_argument("--target_blocks", nargs="+", type=str, default=["up_blocks.0.attentions.1"]) # target blocks to apply IP-Adapter
parser.add_argument("--scale", type=float, default=1.2) # scale for style strength
parser.add_argument("--style_image_id", type=int, default=103)
parser.add_argument("--contet_image_id", type=int, default=14)
args = parser.parse_args()

sdxl_base_model_path = "stabilityai/stable-diffusion-xl-base-1.0" # diffusion model
controlnet_path ="xinsir/controlnet-tile-sdxl-1.0" # tile controlnet for controlling structure of content image
ip_adapter_extractor_path = "IP-Adapter/sdxl_models/image_encoder" # image encoder of IP-Adapter
ip_adapter_module_path = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin" # cross attention module of IP-Adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16).to(device)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    sdxl_base_model_path,
    safety_checker=None,
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_tiling()

model = IPAdapterXL(
    pipe,
    ip_adapter_extractor_path,
    ip_adapter_module_path,
    device,
    target_blocks=args.target_blocks
) # This is the main pipeline we will use for inference
# SDXL diffusion model with both ControlNet and IP-Adapter integrated

del pipe, controlnet
empty_cache() # free up some memory

style_image_path = f'data/style/{args.style_image_id}.jpg' # a style reference image already downloaded in server #[103:Starry Night, #97: Sunflowers, 14: some gogh thing]
content_image_path = f'data/content/{args.contet_image_id}.jpg' # for api server, this is the one we should receive from user upload

style_image = Image.open(style_image_path).convert("RGB") # get image file
content_image = Image.open(content_image_path).convert("RGB") # get image file
H, W = content_image.size # remember original size to resize output later

controlnet_cond_image = resize_image(content_image, short=1024) # resize content image to 1024 short side, because sdxl works better with larger size inputs

input_kwargs = {
    'pil_image': style_image,
    'image': controlnet_cond_image,
} # these are the only thing we need to pass to model, and controlnet_cond_image is from user upload and style_image is from our style reference image database
# ip-adapter will take style_image to extract style features and inject into target blocks during generation
# controlnet will take controlnet_cond_image to guide structure of the generated image

with torch.no_grad():
    generated = model.generate(
        prompt="masterpiece, best quality, high quality",
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        scale=args.scale,
        guidance_scale=5,
        num_samples=1,
        num_inference_steps=30, 
        seed=42,
        controlnet_conditioning_scale=0.6,
        **input_kwargs
    ) # we don't need to change anything else for generation just care about input_kwargs
# it takes 30 steps to get a result, we use ip-adapter scale for style strength control, and keep controlnet strength fixed at 0.6 for controlling structure
os.makedirs(f"results", exist_ok=True)
generated = generated[0].resize((H, W), resample=Image.Resampling.LANCZOS) # resize back to original size
generated.save(f"results/output_{args.style_image_id}_{args.contet_image_id}_{args.scale}.png")    