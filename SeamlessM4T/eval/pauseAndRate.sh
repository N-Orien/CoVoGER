export CUDA_VISIBLE_DEVICES=1
STOPES_ROOT=/mnt/mint/sirou/stopes
GENERATED_DIR=/mnt/mint/sirou/chen/dataeval/best/en2de_emo
GENERATED_TSV=generate-en2de-withSRC.tsv
SRC_LANG=en
TGT_LANG=de

# # src lang pause&rate annotation
# python ${STOPES_ROOT}/stopes/eval/local_prosody/annotate_utterances.py \
#     +data_path=${GENERATED_DIR}/${GENERATED_TSV} \
#     +result_path=${GENERATED_DIR}/${SRC_LANG}_speech_rate_pause_annotation.tsv \
#     +audio_column=src_audio \
#     +text_column=src_text \
#     +speech_units=[syllable] \
#     +vad=true \
#     +net=true \
#     +lang=$SRC_LANG \
#     +forced_aligner=fairseq2_nar_t2u_aligner

# # tgt lang pause&rate annotation
# python ${STOPES_ROOT}/stopes/eval/local_prosody/annotate_utterances.py \
#     +data_path=${GENERATED_DIR}/${GENERATED_TSV} \
#     +result_path=${GENERATED_DIR}/${TGT_LANG}_speech_rate_pause_annotation.tsv \
#     +audio_column=hypo_audio \
#     +text_column=s2t_out \
#     +speech_units=[syllable] \
#     +vad=true \
#     +net=true \
#     +lang=$TGT_LANG \
#     +forced_aligner=fairseq2_nar_t2u_aligner

# pair wise comparison
python ${STOPES_ROOT}/stopes/eval/local_prosody/compare_utterances.py \
    +src_path=${GENERATED_DIR}/${SRC_LANG}_speech_rate_pause_annotation.tsv \
    +tgt_path=${GENERATED_DIR}/${TGT_LANG}_speech_rate_pause_annotation.tsv \
    +result_path=${GENERATED_DIR}/${SRC_LANG}_${TGT_LANG}_pause_scores.tsv \
    +pause_min_duration=0.1