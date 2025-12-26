import torch
import os
import io
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, DDIMScheduler
from ip_adapter import IPAdapterXL
from utils import resize_image, empty_cache

# Force use of GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = FastAPI(title="Gogh Style Transfer API", description="FastAPI wrapper for SDXL + IP-Adapter + ControlNet")

# --- CONFIGURATION ---
SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_PATH = "xinsir/controlnet-tile-sdxl-1.0"
IP_ADAPTER_EXTRACTOR = "IP-Adapter/sdxl_models/image_encoder"
IP_ADAPTER_MODULE = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variable for the model
MODEL = None

@app.on_event("startup")
def load_models():
    global MODEL
    print(f"🚀 Loading models onto {DEVICE.upper()}...")
    
    # 1. Load ControlNet
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float16).to(DEVICE)
    
    # 2. Load Pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_BASE,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(DEVICE)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Optimized for memory/speed without needing xformers
    pipe.vae.enable_tiling()
    pipe.enable_attention_slicing()

    # 3. Load IP-Adapter
    target_blocks = ["up_blocks.0.attentions.1"]
    MODEL = IPAdapterXL(
        pipe,
        IP_ADAPTER_EXTRACTOR,
        IP_ADAPTER_MODULE,
        DEVICE,
        target_blocks=target_blocks
    )
    
    empty_cache()
    print("✅ API System Ready!")

@app.post("/generate-gogh")
async def generate_gogh(
    style_id: int = Form(103),
    scale: float = Form(1.2),
    content_file: UploadFile = File(...)
):
    try:
        # 1. Load Style Image from server
        style_path = f'data/style/{style_id}.jpg'
        if not os.path.exists(style_path):
            raise HTTPException(status_code=404, detail=f"Style image {style_id} not found.")
        style_image = Image.open(style_path).convert("RGB")

        # 2. Process Uploaded Content Image
        content_data = await content_file.read()
        content_image = Image.open(io.BytesIO(content_data)).convert("RGB")
        original_width, original_height = content_image.size
        
        # Prepare for SDXL (ControlNet needs 1024)
        controlnet_cond_image = resize_image(content_image, short=1024)

        # 3. GPU Inference
        print(f"🎨 Generating Gogh style (Scale: {scale})...")
        with torch.no_grad():
            generated = MODEL.generate(
                prompt="masterpiece, best quality, van gogh style, oil painting",
                negative_prompt="text, watermark, lowres, blurry, deformed",
                scale=scale,
                guidance_scale=5.0,
                num_samples=1,
                num_inference_steps=30,
                seed=42,
                controlnet_conditioning_scale=0.6,
                pil_image=style_image,
                image=controlnet_cond_image
            )

        # 4. Save and Return Result
        os.makedirs("results", exist_ok=True)
        res_uuid = str(uuid.uuid4())[:8]
        output_filename = f"results/api_{style_id}_{res_uuid}.png"
        
        # Resize back to original size (Fixing the resample argument)
        final_img = generated[0].resize((original_width, original_height), resample=Image.Resampling.LANCZOS)
        final_img.save(output_filename)

        return FileResponse(output_filename, media_type="image/png")

    except Exception as e:
        print(f"🔥 API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)