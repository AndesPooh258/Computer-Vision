# ENGG5104 Project - Denosing Diffusion Model
#### Dependencies:
1. Python 3.8.10
2. Check codes/setup.py, codes/diffusers/setup.py, codes/diffusers/examples/text_to_image/requirement.txt for more details

#### Code:
1. codes/guided_diffusion/unet.py
    - Python code for Task 1: Complete the UNet of Diffusion Model, Task 3: Class-conditioned Image Generation

2. codes/guided_diffusion/gaussian_diffusion.py
    - Python code for Task 2: Training loss for DDPM

3. codes/scripts/image_sample.py
    - Python codes for image generation in Task 1 - 3

4. codes/diffusers/examples/text_to_image/image_sample.py
    - Python codes for image generation in Task 4

5. codes/diffusers/examples/text_to_image/image_sample_moegirls.py
    - Python codes for image generation in Task 5

#### Commands:
1a. Train a diffusion model on Cable category:
```bash
cd code
bash train.sh
```

1b. Generate images using the trained diffusion model:
```bash
cd code
bash run.sh
```

1c. Evaluation the trained diffusion model:
```bash
cd code
bash eval_nll.sh
```

2a. Train a unified diffusion model on 10 categories:
```bash
cd code
bash train_uni.sh
```

2b. Generate images using the trained unified diffusion model:
```bash
cd code
bash run_uni.sh
```

3a. Perform LoRA training on Pokémon dataset
```bash
cd codes/diffusers/examples/text_to_image
bash train.sh
```

3b. Generate images of Pokémon-like creatures using the trained model
```bash
cd codes/diffusers/examples/text_to_image
python image_sample.py
```

4a. Perform LoRA training on moegirl dataset
```bash
cd codes/diffusers/examples/text_to_image
bash train_moegirls.sh
```

4b. Generate images of moegirls using the trained model
```bash
cd codes/diffusers/examples/text_to_image
python image_sample_moegirls.py
```

#### Outputs:
1. codes/test_cable_single_5k/ema_0.995_005000.pt
    - Trained model for Task 2

2. codes/test_cable_single_5k/log_train.txt
    - Training log for Task 2

3. codes/test_cable_single_5k/log_eval.txt
    - Evaluation log for Task 2

4. codes/test_uni_20k/ema_0.995_020000.pt
    - Trained model for Task 3

5. codes/test_uni_20k/log.txt
    - Training log for Task 3

6. codes/test_run
    - Generated image for Task 2 - 3

7. codes/diffusers/examples/text_to_image/sd-pokemon-model-lora/pytorch_lora_weights.bin
    - Trained LoRA weight for Task 4

8. codes/diffusers/examples/text_to_image/pokemon.png
    - Generated image for Task 4

9. codes/diffusers/examples/text_to_image/sd-moegirl-model-lora/pytorch_lora_weights.bin
    - Trained LoRA weight for Task 5

10. codes/diffusers/examples/text_to_image/moegirls.png
    - Generated image for Task 5

#### Remarks:
1. Modify some lines in codes/scripts/image_sample.py are needed to generate intermediate noisy images
    - Uncomment Line 72 and comment Line 73