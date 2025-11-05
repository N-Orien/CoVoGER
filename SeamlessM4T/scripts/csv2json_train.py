import csv
import json
import sys
import os

def csv_to_json(csv_path, json_path, wav_dir):
    data = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            source_id = int(row[0])
            source_text = row[5]
            target_text = row[6]
            source_audio_path = f"{wav_dir}{source_id}.wav"

            # Check if the audio file exists
            if not os.path.exists(source_audio_path):
                print(f"Skipping entry {source_id} as audio file does not exist.")
                continue

            json_entry = {
                "source": {
                    "id": source_id,
                    "lang": "eng",
                    "text": source_text,
                    "audio_local_path": source_audio_path
                },
                "target": {
                    "id": source_id,
                    "lang": "jpn",
                    "text": target_text
                }
            }

            dup_flag = False
            for i, x in enumerate(data):
                if source_id == x["source"]["id"]:
                    data[i] = json_entry
                    dup_flag = True
                    break
            if dup_flag == False:
                data.append(json_entry)
    
    with open(json_path, 'w', encoding='utf-8') as json_file:
        for entry in data:
            json_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

wav_dir = "/mnt/clover/zd-yang/resources/datasets/EaST-MELD/en-ja/wav/eng/train/"
csv_file_path = "./train/train.csv"
json_file_path = "./train/train.json"
csv_to_json(csv_file_path, json_file_path, wav_dir)
print("Conversion completed successfully!")

