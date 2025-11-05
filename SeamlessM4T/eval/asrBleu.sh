export HYDRA_FULL_ERROR=1
GENERATED_DIR="/mnt/mint/sirou/chen/dataeval/best_DE_DE/de_emo"
SPLIT="en2de"
TGT_LANG="de"

python /mnt/mint/sirou/seamless_communication/src/seamless_communication/cli/expressivity/evaluate/run_asr_bleu.py \
    --generation_dir_path=${GENERATED_DIR} \
    --generate_tsv_filename=generate-${SPLIT}.tsv \
    --tgt_lang=${TGT_LANG}
