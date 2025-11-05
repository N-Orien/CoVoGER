import torch
import sentencepiece as spm
import sys

def convert_lines(input_file_path, output_file_path):
    # Read list from the input file
    data = torch.load(input_file_path)
    new_data = []

    for line in data:
        if line['ground_truth'] != '':
            new_data.append(line)
            
    torch.save(new_data, output_file_path)

#input_file_path = "./test_covost2_en_ja_st_large.pt"
#output_file_path = "./test_covost2_en_ja_st_large.pt.new"

#tgt_lang = "deu"
#translate_lines(input_file_path_deu, output_file_path_deu, translator, src_lang, tgt_lang)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_eos.py input_path output_path")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        convert_lines(input_path, output_path)
