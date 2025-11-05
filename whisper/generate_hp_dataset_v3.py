import whisper
import os, random, copy
import numpy as np
import torch
import pandas as pd
import whisper
import torchaudio
from tqdm.notebook import tqdm
import collections, json
import editdistance
from whisper.normalizers import EnglishTextNormalizer
from num2words import num2words
import re

normalizer = EnglishTextNormalizer()


def calculate_wer(pre, ref):
    return editdistance.eval(pre, ref) / len(ref)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model('large-v3')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HP dataset generation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--asr_wav', type=str, help='wav list file')
    parser.add_argument('--asr_txt', type=str, help='transcription file')
    parser.add_argument('--hp_tsv', type=str, help='generated hp data file')
    parser.add_argument('--beam', type=int)
    args = parser.parse_args()

    f_wav = open(args.asr_wav, 'r')
    f_txt = open(args.asr_txt, 'r')

    json_file = []
    id = 0
    wer = 0

    with open(args.hp_tsv, 'w') as ftsv:
        for line in f_wav.readlines():
            uttid = line.strip().split()[0]
            audio_path = line.strip().split()[-1]
            gt = ' '.join(f_txt.readline().strip().split()[1:])
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            options = whisper.DecodingOptions(language='en', beam_size=args.beam)
            results = whisper.decode(model, mel, options)

            input = []
            for result in results:
                if len(input) < 5 and len(result) > 0 and result not in input:
                    input.append(result)
            if len(input) < 5:
                for _ in range(5 - len(input)):
                    repeat = copy.deepcopy(random.choice(input))
                    input.append(repeat)

            for i in range(len(input)):
                try:
                    text = normalizer(input[i])
                    text = re.sub(r"[-+]?\d*\.?\d+|\d+%?", lambda m: num2words(m.group()), text).replace('%', ' percent')
                except Exception:
                    text = normalizer(input[i])
                    print(f'input exception: {text}')

                input[i] = text if len(text) > 0 else '<UNK>'

            try:
                output = normalizer(gt)
                output = re.sub(r"[-+]?\d*\.?\d+|\d+%?", lambda m: num2words(m.group()), output).replace('%', ' percent')
            except Exception:
                output = normalizer(gt)
                print(f'output exception: {output}')

            output = output if len(output) > 0 else '<UNK>'

    #        data = {"input": input, "output": output}
    #        json_file.append(data)
            ftsv.write(f"{uttid}\t{output}\t{input[0]}\t{input[1]}\t{input[2]}\t{input[3]}\t{input[4]}\n")

            # calculate wer
#            cur_wer = calculate_wer(input[0].split(), output.split())
#            id += 1
#            wer += cur_wer
#            print(f'Utterance {id}: WER = {cur_wer}')
            print(f'Utterance {uttid}')

    f_wav.close()
    f_txt.close()

#    wer /= id
#    print(f'Final WER = {wer}')

#    with open(args.hp_json, 'w') as f:
#        json.dump(json_file, f)

