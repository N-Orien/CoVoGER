#export CUDA_VISIBLE_DEVICES=1

torchrun \
   --rdzv-backend=c10d \
   --rdzv-endpoint=localhost:0 \
   --nnodes=1 \
   --nproc-per-node=1 \
   --no-python \
  m4t_finetune \
   --mode SPEECH_TO_TEXT \
   --train_dataset /mnt/clover/zd-yang/resources/datasets/BMELD/BMELD_data/train_emo.json \
   --eval_dataset /mnt/clover/zd-yang/resources/datasets/BMELD/BMELD_data/dev_emo.json \
   --learning_rate 1e-6\
   --batch_size 4 \
   --warmup_steps 10000 \
   --max_epochs 200 \
   --patience 10 \
   --eval_steps 1000 \
   --model_name seamlessM4T_medium \
   --save_model_to exps/BMELD_medium_emo.pt

