# Moegirl-1.0
This directory consists of the trained model for the ENGG5104 course project, which generates images of moe girls (informally, it means adorable, attractive, cute female characters in anime, manga, video games, and other similar media) based on text prompts.

## Suggested Text Prompts:
Here are some suggested text prompts which could possibly generate realistic images.
```python
prompt = "anime, 1girl, bangs, blonde hair, blush, bookshelf, bow, cowboy shot, green eyes, indoors, long sleeves, looking at viewer, miniskirt, plaid skirt, red skirt, sitting, solo, thighs, white shirt, window"
negative_prompt = "artist name, bad anatomy, bad perspective, bad proportions, blurry, error, extra digits, fewer digits, retro artstyle, signature, simple background, text, username, watermark"
```
```python
prompt = "anime, 1girl, black hair, bare shoulders, blue eyes, blue sky, gradient background, jewelry, looking at viewer, medium hair, necklace, off shoulder, outdoors, small breasts, solo, standing, upper body, white dress"
negative_prompt = "artist name, bad anatomy, bad perspective, bad proportions, blurry, error, extra digits, fewer digits, retro artstyle, signature, simple background, text, username, watermark, medium breasts, large breasts, huge breasts, long hair, very long hair"
```

## Limitations:
As a product of a course project, this model is only trained on an extremely small amount of images (~1000) from the Internet. Therefore, the trained model has several limitations, such as:
- Poor human-like details (e.g., six-fingered hand), deformation, errors, and blurring may appear in the resulting image
- The model may omit some input tags during image generation