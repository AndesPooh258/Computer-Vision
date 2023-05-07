# code reference: https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
import torch
from diffusers import StableDiffusionPipeline

model_path = "./sd-moegirls-model-lora"
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)
pipe.safety_checker = lambda images, clip_input : (images, False)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "anime, 1girl, bangs, blonde hair, blush, bookshelf, bow, cowboy shot, green eyes, indoors, long sleeves, looking at viewer, miniskirt, plaid skirt, red skirt, sitting, solo, thighs, white shirt, window"
negative_prompt = "artist name, bad anatomy, bad perspective, bad proportions, blurry, error, extra digits, fewer digits, retro artstyle, signature, simple background, text, username, watermark"

'''
prompt = "anime, 1girl, black hair, bare shoulders, blue eyes, blue sky, gradient background, jewelry, looking at viewer, medium hair, necklace, off shoulder, outdoors, small breasts, solo, standing, upper body, white dress"
negative_prompt = "artist name, bad anatomy, bad perspective, bad proportions, blurry, error, extra digits, fewer digits, retro artstyle, signature, simple background, text, username, watermark, medium breasts, large breasts, huge breasts, long hair, very long hair"
'''

image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("moegirls.png")
