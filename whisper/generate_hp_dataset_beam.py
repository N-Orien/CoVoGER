import whisper
import os, random, copy
import numpy as np
import torch
import pandas as pd
import torchaudio
from tqdm.notebook import tqdm
import collections, json
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HP dataset generation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--asr_tsv', type=str, help='tsv file')
    parser.add_argument('--hp_tsv', type=str, help='generated hp data file')
    parser.add_argument('--beam', type=int)
    parser.add_argument('--language', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    model = whisper.load_model(args.model)
    json_file = []
    id = 0
    wer = 0

    with open(args.asr_tsv, 'r') as f_input, open(args.hp_tsv, 'w') as f_output:
        for line in f_input.readlines():
            uttid, audio_path, gt = line.strip().split('\t')
#            audio = whisper.load_audio(audio_path)
#            audio = whisper.pad_or_trim(audio)
#            mel = whisper.log_mel_spectrogram(audio).to(model.device)
#            options = whisper.DecodingOptions(language=args.language, temperature=args.temperature, best_of=5)
#            options = whisper.DecodingOptions(language=args.language, beam_size=5)
#            results = whisper.decode(model, mel, options)

            input = []
            results = model.transcribe(audio_path, language=args.language, beam_size=args.beam)
#            print(results)
            input.append(results['text'])

#            f_output.write(f"{uttid}\t{gt}\t{input[0]}\t{input[1]}\t{input[2]}\t{input[3]}\t{input[4]}\n")
            f_output.write(f"{uttid}\t{gt}\t{input[0]}\n")
            
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"Processed {uttid} at {current_time}")


