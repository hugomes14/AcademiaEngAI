# pip install -U diffusers

import torch
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.float16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
pipe.enable_attention_slicing()

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=1.0,
    num_inference_steps=2,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]
image.save("flux2-img.png")
