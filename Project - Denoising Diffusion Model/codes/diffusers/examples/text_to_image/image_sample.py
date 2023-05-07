# code reference: https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
import torch
from diffusers import StableDiffusionPipeline

model_path = "./sd-pokemon-model-lora"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.safety_checker = lambda images, clip_input : (images, False)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "cute dragon creature."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")