#!/usr/bin/env bash

dataset=bmeld
srclang=en
tgtlang=zh
task=st
seamless_size=medium
method=emotion_output_conv1d
data_dir=./example_data/${seamless_size}_${method}
llm_dir=./GenTranslate/checkpoints/meta-llama/Llama-2-7b-hf

#python finetune/gentrans_conv1d.py \	#if using 1-D convolutional modality adapter
#python finetune/gentrans_qformer.py \	#if using Q-Former modality adapter
python finetune/gentrans.py \	#if not using modality adapter
       --dataset ${dataset} --srclang ${srclang} --tgtlang ${tgtlang} --task ${task} --d 1 \
       --seamless_size ${seamless_size} --data_dir ${data_dir} --llm_dir ${llm_dir}  \
       --lr 0.01 --num_epochs 2 --name ${dataset}_${seamless_size}_${method}

