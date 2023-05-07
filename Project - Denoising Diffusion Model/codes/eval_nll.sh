MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm False"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"

# specify the experiment dir
dirname='./test_cable_single_5k'
export OPENAI_LOGDIR=$dirname

model=$dirname'/ema_0.995_005000.pt'
python ./scripts/image_nll.py --data_dir ./data/MVTecAD/cable/  --model_path $model \
		$MODEL_FLAGS $DIFFUSION_FLAGS  --class_cond False --num_sample 100 --timestep_respacing 250