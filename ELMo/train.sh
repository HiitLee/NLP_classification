
python -m biLM train \
	 --train_path data/original2.txt \
	 --config_path configs/cnn_50_100_512_4096_sample.json \
	 --model output/en \
	 --optimizer adam \
	 --gpu 1 \
	 --lr 0.001 \
	 --lr_decay 0.8 \
	 --max_epoch 1 \
	 --max_sent_len 20 \
	 --max_vocab_size 150000 \
	 --min_count 3
