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

speech_length = []
text_length = []

#for i in range(10):
for line in data:
    speech_length.append(int(line["encoder_output"].shape[1]))
    text_length.append(int(line["input_ids"].shape[0]))

data_len = len(data)

text_over512 = sum(1 for l in text_length if l > 512)
text_over1024 = sum(1 for l in text_length if l > 1024)
text_over2048 = sum(1 for l in text_length if l > 2048)

print("Data size:", data_len)
print("Longest speech:", max(speech_length))
print("text over 512", text_over512)
print("text over 2048", text_over2048)
