import os
import csv

def translate_lines(input_file_path_csv, wav_dir):
    # Read lines from the input file
    with open(input_file_path_csv, "r", encoding="gbk") as input_csv:
        csv_reader = csv.reader(input_csv)

        for row in csv_reader:
            dia_id = row[5]
            utt_id = row[6]
            dia_utt = f"dia{dia_id}_utt{utt_id}"
            audio_path = wav_dir + f"{dia_utt}.wav"
            if not os.path.exists(audio_path):
                print(f"no wav {audio_path}")

setname = "test"

wav_dir = f"/mnt/clover/zd-yang/resources/datasets/MELD/audio/{setname}/"
input_file_path_csv = f"/mnt/clover/zd-yang/resources/datasets/BMELD/BMELD_data/{setname}_sent_emo.csv"

translate_lines(input_file_path_csv, wav_dir)

