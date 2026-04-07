import json
from litgpt import LLM
from litgpt.prompts import Alpaca
import sys
import torch

data_path = sys.argv[1]
exp_dir = sys.argv[2]
checkpoint = sys.argv[3]

# ── 1) Point to your LoRA checkpoint directory ───────────────────────────────
# This folder must contain the 'lit_model.pth.lora' file produced by `litgpt finetune_lora`
model_dir = f"{exp_dir}/{checkpoint}"  
pred_path = f"{model_dir}/pred.txt"

# ── 2) Load the LoRA-tuned model once ────────────────────────────────────────
# LitGPT will detect the '.lora' file and merge adapters into the base model in memory
llm = LLM.load(model_dir)  
llm.model.eval()

# instantiate the same prompt style you used when training
prompt_style = Alpaca()

# ── 3) Read your JSON array and batch-infer ─────────────────────────────────
with open(data_path, "r", encoding="utf-8") as f:
    records = json.load(f)

with open(pred_path, "w", encoding="utf-8") as out_f:
    with torch.no_grad():
        for i, rec in enumerate(records):
            instr = rec["instruction"].strip()
            inp = rec["input"].strip()
            prompt = prompt_style.apply(prompt=instr, input=inp)
            pred = llm.generate(
                prompt,
                max_new_tokens=256,
                top_k=1,
            ).strip()
            pred = pred.replace("\n", "#")
            out_f.write(pred + "\n")
