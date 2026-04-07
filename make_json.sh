#!/bin/bash

COVOGER_DIR=$1

mkdir -p json/asr
for model in small medium large; do
for decode in beam sample mix; do
  mkdir -p json/asr/all_${model}_${decode}
for lang in ar ca cy de en et fa id ja lv sl sv ta tr zh; do
for split in train dev test; do
	python make_json_asr.py ${COVOGER_DIR}/asr/${lang}/${split}_${model}_${decode}.tsv -o json/asr/${lang}_${model}_${decode}/${split}.json
	cat json/asr/${lang}_${model}_${decode}/${split}.json >> json/asr/all_${model}_${decode}/${split}.json
done
done
done
done

mkdir -p json/st
for model in medium large; do
for decode in beam sample mix; do
  mkdir -p json/st/all_${model}_${decode}
for lang in ar ca cy de et fa id ja lv sl sv ta tr zh; do
for langpair in ${lang}-en en-${lang}; do
for split in train dev test; do
	python make_json_st.py ${COVOGER_DIR}/st/${langpair}/${split}_${model}_${decode}.tsv -o json/st/${langpair}_${model}_${decode}/${split}.json
	cat json/st/${langpair}_${model}_${decode}/${split}.json >> json/st/all_${model}_${decode}/${split}.json
done
done
done
done
done

rename dev val json/*/*/*
