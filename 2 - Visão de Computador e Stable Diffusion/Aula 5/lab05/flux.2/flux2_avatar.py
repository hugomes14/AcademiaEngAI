# pip install -U diffusers

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.float16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
pipe.enable_attention_slicing()

prompt = "Make an avatar"
image_path = 'ronaldo.jpg'
 
init_image = Image.open(image_path).convert("RGB")

image = pipe(
    image=init_image,
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=25,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]
image.save("flux2-avatar.png")
