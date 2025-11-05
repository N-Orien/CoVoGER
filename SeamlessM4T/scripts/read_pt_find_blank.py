import sys
import torch
import sentencepiece as spm

if len(sys.argv) != 2:
    print("Usage: python convert_pt.py <filename>")
    sys.exit(1)

file_path = sys.argv[1]

# Load the provided dataset file
data = torch.load(file_path)

# Display the structure of the first entry in the dataset

sp = spm.SentencePieceProcessor(model_file='/mnt/clover/zd-yang/works/EmoST_LLM/GenTranslate/checkpoints/meta-llama/Llama-2-7b-hf/tokenizer.model')

for line in data:
    if len(line['ground_truth']) < 1:
        print(line['ground_truth'])
        for key, value in line.items():
            print(f"{key}: {value}")

print(len(data))
