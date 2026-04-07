#!/bin/bash

DATADIR=

for lang in ar ca cy de en et fa id ja lv sl sv ta tr zh; do
for model in small medium large; do
for decode in beam sample mix; do
for split in train dev test; do
	python make_json_asr.py ${DATADIR}/asr/${lang}/${split}_${model}_${decode}.tsv -o json/asr/${lang}_${model}_${decode}/${split}.json
done
done
done
done

for lang in ar ca cy de et fa id ja lv sl sv ta tr zh; do
for langpair in ${lang}-en en-${lang}; do
for model in medium large; do
for decode in beam sample mix; do
for split in train dev test; do
	python make_json_st.py ${DATADIR}/st/${langpair}/${split}_${model}_${decode}.tsv -o json/st/${langpair}_${model}_${decode}/${split}.json
done
done
done
done
done
