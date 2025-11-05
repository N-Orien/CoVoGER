import csv
import json
import sys

def csv_to_json(csv_path, json_path, wav_dir):
    data = []
    with open(csv_path, mode='r', encoding='gbk') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[-3] == 'Target':
                continue
            source_id = f"dia{row[5]}_utt{row[6]}"
            source_text = row[1].strip('\"')
            target_text = row[-3]
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

setname = 'train'

wav_dir = f"/mnt/clover/zd-yang/resources/datasets/MELD/audio/{setname}/"
csv_file_path = f"/mnt/clover/zd-yang/resources/datasets/BMELD/BMELD_data/{setname}_sent_emo_filtered.csv"
json_file_path = f"/mnt/clover/zd-yang/resources/datasets/BMELD/BMELD_data/{setname}.json"
csv_to_json(csv_file_path, json_file_path, wav_dir)
print("Conversion completed successfully!")

