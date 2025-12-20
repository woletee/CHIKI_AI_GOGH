import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
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

sdxl_base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_path ="diffusers/controlnet-canny-sdxl-1.0"
ip_adapter_extractor_path = "IP-Adapter/sdxl_models/image_encoder"
ip_adapter_module_path = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

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
    target_blocks=["up_blocks.0.attentions.1"]
) # This is the main pipeline we will use for inference

del pipe, controlnet
empty_cache()

style_image_path = 'data/style/103.jpg' # a style reference image already downloaded in server #[103:Starry Night, #97: Sunflowers, 14: some gogh thing]
content_image_path = 'data/content/14.jpg' # for api server, this is the one we should receive from user upload

style_image = Image.open(style_image_path).convert("RGB")
content_image = Image.open(content_image_path).convert("RGB")
H, W = content_image.size # remember original size to resize output later

resized_image = resize_image(content_image, short=1024)
resized_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
detected_map = cv2.Canny(resized_image, 50, 200)
canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
controlnet_cond_image = canny_map # this aims to preserve content structure from user upload

input_kwargs = {
    'pil_image': style_image,
    'image': controlnet_cond_image,
} # these are the only thing we need to pass to model, and controlnet_cond_image is from user upload and style_image is from our style reference image database

with torch.no_grad():
    generated = model.generate(
        prompt="masterpiece, best quality, high quality",
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        scale=1.0,
        guidance_scale=5,
        num_samples=1,
        num_inference_steps=30, 
        seed=42,
        controlnet_conditioning_scale=0.6,
        **input_kwargs
    ) # we don't need to change anything else for generation just care about input_kwargs
generated = generated[0].resize((H, W), resample=Image.Resampling.LANCZOS)
generated.save("output.png")    