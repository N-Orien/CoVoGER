import torch
from transformers import PreTrainedTokenizerFast
import sys

def convert_lines(input_file_path, output_file_path):
    # Read list from the input file
    data = torch.load(input_file_path)
    new_data = []
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="/mnt/clover/zd-yang/works/EmoST_LLM/GenTranslate/checkpoints/meta-llama/Meta-Llama-3.1-8B/tokenizer.json")
    eos = torch.tensor([127999], dtype=torch.int32)

    for line in data:
        hyps = line['input']
        ground_truth = line['ground_truth']

        inputs = f"Below is the best-hypotheses transcribed from speech translation system. Please try to revise it using the words which are only included into other-hypothesis, and write the response for the true transcription.\n\n### Best-hypothesis:\n{hyps[0]}.\n\n### Other-hypothesis:\n{hyps[1]}. {hyps[2]}. {hyps[3]}. {hyps[4]}.\n\n### Response:\n{ground_truth}"
        inputs_no_response = f"Below is the best-hypotheses transcribed from speech translation system. Please try to revise it using the words which are only included into other-hypothesis, and write the response for the true transcription.\n\n### Best-hypothesis:\n{hyps[0]}.\n\n### Other-hypothesis:\n{hyps[1]}. {hyps[2]}. {hyps[3]}. {hyps[4]}.\n\n### Response:\n"

        input_ids = tokenizer(inputs)["input_ids"]
        input_ids = input_ids[1:]
        input_ids_no_response = tokenizer(inputs_no_response)["input_ids"]
        input_ids_no_response = input_ids_no_response[1:]
        labels = [-100 if i < len(input_ids_no_response) else input_ids[i] for i in range(len(input_ids))]

        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        input_ids_no_response = torch.tensor(input_ids_no_response, dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int32)

        input_ids = torch.cat((input_ids, eos))
        labels = torch.cat((labels, eos))

        line["input_ids"] = input_ids
        line["input_ids_no_response"] = input_ids_no_response
        line["labels"] = labels

        new_data.append(line)

    torch.save(new_data, output_file_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_data.py input_path output_path")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        convert_lines(input_path, output_path)
