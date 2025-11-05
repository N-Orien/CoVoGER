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

    emotion_dict = {}
    for _, row in csv_data.iterrows():
        dia_utt = f"dia{row[5]}_utt{row[6]}"
        emotion_dict[dia_utt] = (row[3], row[4])
#        emotion_dict[str(row[0])] = (row[3], row[4])
        
    for line in pt_data:
        hyps = line['input']
        ground_truth = line['ground_truth']
        index = str(line['id'])
        if index in emotion_dict:
            emotion, sentiment = emotion_dict[index]

            inputs = f"You will be shown the best-hypotheses transcribed from speech translation system. Please try to predict the emotion and the sentiment of the speech, and try to revise the best-hypothesis using the words which are included in other-hypothesis. Please write the response in the following format:\nEmotion\nSentiment\nTrue transcription.\n\n### Best-hypothesis:\n{hyps[0]}.\n\n### Other-hypothesis:\n{hyps[1]}. {hyps[2]}. {hyps[3]}. {hyps[4]}.\n\n### Response:\n{emotion}\n{sentiment}\n{ground_truth}"
            inputs_no_response = f"You will be shown the best-hypotheses transcribed from speech translation system. Please try to predict the emotion and the sentiment of the speech, and try to revise the best-hypothesis using the words which are included in other-hypothesis. Please write the response in the following format:\nEmotion\nSentiment\nTrue transcription.\n\n### Best-hypothesis:\n{hyps[0]}.\n\n### Other-hypothesis:\n{hyps[1]}. {hyps[2]}. {hyps[3]}. {hyps[4]}.\n\n### Response:\n"

            input_ids = sp.EncodeAsIds(inputs)
            input_ids_no_response = sp.EncodeAsIds(inputs_no_response)
            labels = [-1 if i < len(input_ids_no_response) else input_ids[i] for i in range(len(input_ids))]


            input_ids = torch.tensor(input_ids, dtype=torch.int32)
            input_ids_no_response = torch.tensor(input_ids_no_response, dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int32)

            input_ids = torch.cat((input_ids, eos))
            labels = torch.cat((labels, eos))

            line["input_ids"] = input_ids
            line["input_ids_no_response"] = input_ids_no_response
            line["labels"] = labels

            new_data.append(line)
        
        else:
            print(f"{output_file_path}")
            print(f"ERROR: {index}")

    torch.save(new_data, output_file_path)

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
