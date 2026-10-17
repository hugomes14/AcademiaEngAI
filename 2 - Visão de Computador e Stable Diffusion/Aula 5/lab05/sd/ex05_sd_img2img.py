# pip install -U diffusers

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    'stablediffusionapi/deliberate-v2',
    torch_dtype=torch.float16
).to("cuda:0")
 
image_path = 'abstract_art.jpg'
 
init_image = Image.open(image_path).convert("RGB")
init_image = init_image.resize((768, 768))

prompt = "A fantasy landscape, with starry night sky, moonlit river, trending on artstation"
seed=6
 
image = pipe(
    prompt=prompt,
    image=init_image,
    num_inference_steps=150,
    strength=1,
    generator=torch.manual_seed(seed)
).images[0]

image.save("imagem_ex5.jpg")