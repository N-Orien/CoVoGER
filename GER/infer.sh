#!/usr/bin/env bash

#source activate <your-conda-env>

### Note: "Llama-2-7b-hf" for x-en, and "Llama-2-13b-hf" for en-x;
dataset=bmeld
srclang=en
tgtlang=zh
task=st
seamless_size=medium
method=emotion_output_conv1d
data_dir=./example_data/${seamless_size}_${method}
llm_dir=./checkpoints/meta-llama/Llama-2-7b-hf
adapter_path=./runs/${dataset}_${seamless_size}_${method}/best_adapter.pth

#python inference/gentrans_conv1d.py \	#if using emotin labels as outputs & using 1-D convolutional modality adapter
#python inference/gentrans_qformer.py \	#if using emotin labels as outputs & using Q-Former modality adapter
#python inference/gentrans_emoout.py \	#if using emotin labels as outputs & not using modality adapter
python inference/gentrans.py \	#if not using emotin labels as outputs & not using modality adapter
        --dataset ${dataset} --srclang ${srclang} --tgtlang ${tgtlang} --task ${task} \
        --seamless_size ${seamless_size} --data_dir ${data_dir} --llm_dir ${llm_dir} \
        --adapter_path ${adapter_path}
