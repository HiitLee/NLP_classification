#! /bin/sh

python main.py --lr  1e-4 \
	--epochs 1 \
	--batch 256 \
	--maxSen 100 \
	--model CNN \
	--mode train \
	--sampling True \
	--tokenizer char \
	--data_train_file ./nsmc/ratings_train.tsv \
	--data_test_file ./nsmc/ratings_test.tsv \
	--vocab 500000 

