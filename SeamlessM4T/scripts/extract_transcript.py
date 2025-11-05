import torch
import sentencepiece as spm
import sys
import pandas as pd

def convert_lines(input_file_path, emotion_file_path, output_file_path):
    # Read list from the input file
    pt_data = torch.load(input_file_path)
    new_data = []
    sp = spm.SentencePieceProcessor(model_file='/mnt/clover/zd-yang/works/EmoST_LLM/GenTranslate/checkpoints/meta-llama/Llama-2-7b-hf/tokenizer.model')
    eos = torch.tensor([2], dtype=torch.int32)
    csv_data = pd.read_csv(emotion_file_path, encoding='gbk')

    asr_dict = {}
    for _, row in csv_data.iterrows():
        dia_utt = f"dia{row[5]}_utt{row[6]}"
        asr_dict[dia_utt] = row[1]
#        emotion_dict[str(row[0])] = (row[3], row[4])

    transcript_list = []
    for line in pt_data:
        hyps = line['input']
        ground_truth = line['ground_truth']
        index = str(line['id'])
        if index in asr_dict:
            transcript = asr_dict[index]
            transcript_list.append(transcript)
        else:
            print(f"{output_file_path}")
            print(f"ERROR: {index}")

    with open(output_file_path, 'w') as fo:
        for transcript in transcript_list:
            fo.write(f"{transcript}\n")

#input_file_path = "/mnt/clover/zd-yang/resources/datasets/EaST-MELD/en-ja/hyps/dev_subtitle/text.pt.large"
#emotion_file_path = "/mnt/clover/zd-yang/resources/datasets/EaST-MELD/en-ja/txt/dev_subtitle/dev.csv"
#output_file_path = "/mnt/clover/zd-yang/resources/datasets/EaST-MELD/en-ja/hyps/dev_east-meld_en_ja_st_large_emotion.pt"

#tgt_lang = "deu"
#translate_lines(input_file_path_deu, output_file_path_deu, translator, src_lang, tgt_lang)
#convert_lines(input_file_path, emotion_file_path, output_file_path)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python convert_data.py input_path emotion_path output_path")
    else:
        input_path = sys.argv[1]
        emotion_path = sys.argv[2]
        output_path = sys.argv[3]
        convert_lines(input_path, emotion_path, output_path)
