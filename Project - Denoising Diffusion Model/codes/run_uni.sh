MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"

# specify the experiment dir
dirname='./test_uni_20k'
model=$dirname'/ema_0.995_020000.pt'

# log dir
export OPENAI_LOGDIR=test_run
python ./scripts/image_sample.py --model_path $model \
	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 250 --num_samples 30 --class_cond True