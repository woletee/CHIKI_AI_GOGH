import torch
import gc
from PIL import Image

def resize_image(image: Image.Image, short=512, mult=16):
    w, h = image.size

    if w<=h:
        new_w = short
        new_h = int(round(h * (short/w)))
        new_h = int(round(new_h / mult)) * mult
    else:
        new_h = short
        new_w = int(round(w * (short / h)))
        new_w = int(round(new_w / mult)) * mult
    
    return image.resize((new_w, new_h), resample=Image.BICUBIC)

def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()