# CoVoGER

Code for the paper **[CoVoGER: A Multilingual Multitask Benchmark for Speech-to-text Generative Error Correction with Large Language Models](https://aclanthology.org/2025.emnlp-main.320/)**.

The CoVoGER dataset is available on [Hugging Face](https://huggingface.co/datasets/PeacefulData/CoVoGER).

## 1. Build the Environment

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate covoger
```

## 2. Data Preparation

After downloading the CoVoGER dataset, construct the JSON files with instruction prompts and the files that combine all languages:

```bash
bash make_json.sh ${COVOGER_DIR}
```

- `${COVOGER_DIR}`: the root directory of CoVoGER

## 3. Training

Run fine-tuning with:

```bash
bash finetune.sh ${MODEL_DIR} ${DATA_DIR} ${EXP_DIR}
```

- `${MODEL_DIR}`: the directory of the LLM downloaded with `litgpt`  
  Example:
  ```bash
  litgpt download Qwen/Qwen3-8B-Base
  ```
- `${DATA_DIR}`: the directory of the training data  
  Example: `json/asr/all_large_beam`
- `${EXP_DIR}`: the directory to save checkpoints

## 4. Inference

Run inference with:

```bash
python inference.py ${DATA_DIR} ${EXP_DIR} ${CHECKPOINT}
```

- `${DATA_DIR}`: the test data file  
  Example: `json/asr/en_large_beam/test.json`
- `${EXP_DIR}`: the directory of saved checkpoints (same as in training)
- `${CHECKPOINT}`: the checkpoint to use for inference  
  Example: `final`
