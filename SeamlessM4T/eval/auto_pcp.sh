
export CUDA_VISIBLE_DEVICES=1
python -m stopes.modules +compare_audios=AutoPCP_multilingual_v2 \
    launcher.cluster=local \
    +compare_audios.input_file=/mnt/mint/sirou/chen/dataeval/best_DE_DE/de_noEmo/generate-en2de.tsv \
    compare_audios.src_audio_column=src_audio \
    compare_audios.tgt_audio_column=hypo_audio \
    +compare_audios.named_columns=true \
    +compare_audios.output_file=/mnt/mint/sirou/chen/dataeval/best_DE_DE/de_noEmo/autopcp_output.txt