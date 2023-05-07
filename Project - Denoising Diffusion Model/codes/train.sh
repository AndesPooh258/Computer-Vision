MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm False --use_fp16 False"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --save_interval 2500 --lr_anneal_steps 5000 --ema_rate 0.995"

# specify the experiment dir
dirname='./test_cable_single_5k'
export OPENAI_LOGDIR=$dirname

python ./scripts/image_train.py --data_dir ./data/MVTecAD/cable/ \
		$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --class_cond False
