export HYDRA_FULL_ERROR=1

SPEECH_ENCODER_MODEL_PATH=/mnt/mint/sirou/UniSpeech/wavlm_large_finetune.pth
INPUT_PATH=/mnt/mint/sirou/chen/dataeval/best_DE_DE/de_emo/generate-en2de-withSRC.tsv
RESULT_PATH=/mnt/mint/sirou/chen/dataeval/best_DE_DE/de_emo/

python -m stopes.modules \
    +vocal_style_similarity=base \
    launcher.cluster=local \
    vocal_style_similarity.model_type=valle \
    +vocal_style_similarity.model_path=${SPEECH_ENCODER_MODEL_PATH} \
    +vocal_style_similarity.input_file=${INPUT_PATH} \
    +vocal_style_similarity.output_file=${RESULT_PATH}/vocal_style_sim_result.txt \
    vocal_style_similarity.named_columns=true \
    vocal_style_similarity.src_audio_column=src_audio \
    vocal_style_similarity.tgt_audio_column=hypo_audio