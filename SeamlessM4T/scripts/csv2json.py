import csv
import json
import sys

def csv_to_json(csv_path, json_path, wav_dir):
    data = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] == 'ID':
                continue
            source_id = int(row[0])
            source_text = row[5].strip('\"')
            target_text = row[6]
            source_audio_path = f"{wav_dir}{source_id}.wav"
            
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
            data.append(json_entry)
    
    with open(json_path, 'w', encoding='utf-8') as json_file:
        for entry in data:
            json_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

wav_dir = "/mnt/clover/zd-yang/resources/datasets/EaST-MELD/en-ja/audio/eng/train/train_"
csv_file_path = "/mnt/clover/zd-yang/resources/datasets/EaST-MELD/en-ja/ENG_JPN/train.csv"
json_file_path = "/mnt/clover/zd-yang/resources/datasets/EaST-MELD/en-ja/ENG_JPN/train.json"
csv_to_json(csv_file_path, json_file_path, wav_dir)
print("Conversion completed successfully!")

