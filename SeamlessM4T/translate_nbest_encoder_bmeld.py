import torch
from seamless_communication.inference import Translator
import csv
import sys

WAV_DIR = "/mnt/clover/zd-yang/resources/datasets/MELD/audio"
TEXT_DIR = "./example_data/BMELD"
OUTPUT_DIR = "./example_data/output"

def translate_lines(input_file_path_csv, output_file_path, wav_dir, translator, src_lang, tgt_lang):
    # Read lines from the input file
    with open(input_file_path_csv, "r", encoding="gbk") as input_csv:
        csv_reader = csv.reader(input_csv)

        # Translate each line and save the results
        output_list = []
        for row in csv_reader:
            try:
                dia_id = row[5]
                utt_id = row[6]
                dia_utt = f"dia{dia_id}_utt{utt_id}"
                audio_path = wav_dir + f"{dia_utt}.wav"
                translated_texts, _, encoder_output = translator.predict(audio_path, "S2TT", tgt_lang=tgt_lang, src_lang=src_lang, n_best=True)
                ground_truth = row[11]

                utt_dict={}
                utt_dict["id"] = dia_utt
                translated_texts = [str(c) for c in translated_texts[0]]
                utt_dict["input"] = translated_texts
                utt_dict["encoder_output"] = encoder_output
                utt_dict["ground_truth"] = ground_truth
                print(utt_dict["id"])
                output_list.append(utt_dict)
            except:
                print("No wav file")
        torch.save(output_list, output_file_path)

setname = sys.argv[1]
modelsize = sys.argv[2]

src_lang = "eng"
tgt_lang = "cmn"

translator = Translator(f"seamlessM4T_{modelsize}", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"), dtype=torch.float32) 

wav_dir = f"{WAV_DIR}/{setname}/"
input_file_path_csv = f"{TEXT_DIR}/{setname}_sent_emo.csv"
output_file_path = f"{OUTPUT_DIR}/{setname}_{modelsize}.pt"

translate_lines(input_file_path_csv, output_file_path, wav_dir, translator, src_lang, tgt_lang)

