MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --save_interval 10000 --lr_anneal_steps 20000 --ema_rate 0.995"

# specify the experiment dir
dirname='./test_uni_20k'
export OPENAI_LOGDIR=$dirname

python ./scripts/image_train.py --data_dir ./data/MVTecAD/ \
		$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --class_cond True
