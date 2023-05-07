export MODEL_NAME="stabilityai/stable-diffusion-2"
export DATASET_NAME="./data/moegirls"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=768 --center_crop --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-moegirls-model-lora" \
  --validation_prompt="blonde hair, green eyes, long hair, white dress"