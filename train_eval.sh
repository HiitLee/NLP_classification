
#! /bin/sh


python main.py \
	--model LSTM \
	--mode train \
	--batch 256 \
	--vocab 500000 \
	--maxSen 5 \
	--sampling True \
	--tokenizer word \
	--lr 1e-3 \
	--data_train_file ./nsmc/ratings_train.tsv \
	--data_test_file ./nsmc/ratings_test.tsv 

