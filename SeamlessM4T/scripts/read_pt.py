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

#for i in range(10):
for i, line in enumerate(data):
    if i == 245 or i ==296 or i==292 or i==322 or i==240 or i==396:
        for key, value in line.items():
            if key == 'encoder_output':
                print("SHAPE", value.shape)
            print(f"{key}: {value}")
#    input_ids = line['input_ids']
#    no_response = data[i]['input_ids_no_response']
#    decoded_text = sp.decode_ids(input_ids.tolist())
#    decoded_text2 = sp.decode_ids(no_response.tolist())
#    print(decoded_text)
#    print(decoded_text2)

print(len(data))
