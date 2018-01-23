#!/bin/bash -e
mkdir -p gen_data
./gen3.py
rm -rf ./sched_model
(cd sequence_tagging
 python build_data.py
 python train.py)
exec python -m nmt.nmt.nmt    \
	--attention=scaled_luong \
       	--src=en --tgt=js \
     	--vocab_prefix=./gen_data/vocab \
   	--train_prefix=./gen_data/train \
    	--dev_prefix=./gen_data/dev \
   	--test_prefix=./gen_data/tst \
    	--out_dir=./sched_model \
    	--num_train_steps=12000 \
    	--steps_per_stats=100 \
    	--num_layers=2 \
    	--num_units=128 \
    	--dropout=0.2 --metrics=bleu
