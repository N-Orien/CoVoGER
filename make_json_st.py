#!/usr/bin/env python3
import argparse
import json
import sys
import os
from pathlib import Path

LANG_MAP = {
    'en': 'English',
    'ar': 'Arabic',
    'ca': 'Catalan',
    'cy': 'Welsh',
    'de': 'German',
    'et': 'Estonian',
    'fa': 'Persian',
    'id': 'Indonesian',
    'ja': 'Japanese',
    'lv': 'Latvian',
    'sl': 'Slovenian',
    'sv': 'Swedish',
    'ta': 'Tamil',
    'tr': 'Turkish',
    'zh': 'Chinese'
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an N-best TSV into JSON for LLM training."
    )
    parser.add_argument(
        "tsv_file", help="Input TSV file (ID + ground truth + N-best hypotheses)"
    )
    parser.add_argument(
        "-o", "--output", dest="json_file", required=True,
        help="Output JSON file"
    )
    return parser.parse_args()


def detect_language(tsv_path):
    """
    Detect language by taking the second-to-last path segment as a 2-letter code.
    Defaults to English if the segment isn't in LANG_MAP or path too short.
    """
    parts = os.path.normpath(tsv_path).split(os.sep)
    if len(parts) >= 2:
        code = parts[-2]
        src, tgt = code.split('-')
        return LANG_MAP[src], LANG_MAP[tgt]


def main():
    args = parse_args()

    Path(args.json_file).parent.mkdir(parents=True, exist_ok=True)

    # Determine language for instruction
    src, tgt = detect_language(args.tsv_file)
    instruction = (
        f"You are given the N-best {tgt} translations for the same {src} speech segment.\n"
        "Your task is to output one corrected translation that is most likely to match the speech."
    )
    examples = []

    with open(args.tsv_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) < 3:
                print(
                    f"Skipping malformed line (need ≥3 cols): {line}",
                    file=sys.stderr
                )
                continue

            # parts[0] is the utterance ID, ignore
            ground_truth = parts[1]
            nbest_list = parts[2:]
            input_text = "\n".join(nbest_list)

            examples.append({
                "instruction": instruction,
                "input": input_text,
                "output": ground_truth
            })

    with open(args.json_file, "w", encoding="utf-8") as out_f:
        json.dump(examples, out_f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

