# pip install -U diffusers
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2"
    , torch_dtype = torch.float16
).to("cuda:0")

prompt = "a photograph of an astronaut riding a horse"
negative_prompt = "smooth, neon, 3D render"
image = pipe(
    prompt          = prompt
    , negative_prompt=negative_prompt
    , generator     = torch.Generator("cuda:0").manual_seed(6)
    , width         = 768
    , height        = 512
).images[0]

image.save("imagem_ex3.png")