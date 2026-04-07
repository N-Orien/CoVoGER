#!/bin/bash

ref=$1
hyp=$2
langpair=$3

IFS='-' read -r -a parts <<< "$langpair"
if [[ "${parts[1]}" == "ja" ]]; then
	sacrebleu ${ref} -i ${hyp} -m bleu -tok ja-mecab -b --width 2
elif [[ "${parts[1]}" == "zh" ]]; then
	sacrebleu ${ref} -i ${hyp} -m bleu -tok zh -b --width 2 
else
	sacrebleu ${ref} -i ${hyp} -m bleu -b --width 2
fi
