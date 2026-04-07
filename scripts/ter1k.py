import sys
from jiwer import wer
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh


def calculate_ter(ref_file, hyp_file, lang):
    with open(ref_file, 'r', newline='', encoding='utf-8') as reffile, open(hyp_file, 'r', newline='', encoding='utf-8') as hypfile:
        reflines = reffile.readlines()
        hyplines = hypfile.readlines()
        reflines = reflines[:1000]
        hyplines = hyplines[:1000]
        lines = zip(reflines, hyplines)

        if lang == "ja":
            tokenizer = TokenizerJaMecab()
        elif lang == "zh":
            tokenizer = TokenizerZh()
        else:
            tokenizer = Tokenizer13a()

        total_wer = 0
        num_lines = 0

        for ref, hyp in lines:
            reference = tokenizer(ref.strip())
            hypothesis = tokenizer(hyp.strip())

            current_wer = wer(reference, hypothesis)
            total_wer += current_wer

            num_lines += 1

        # Averages
        average_wer = total_wer / num_lines

        return average_wer
            
def print_wer(wer_list):
    output = "("
    for wer in wer_list:
        wer_str = f"{wer * 100:.1f}, "
        output += wer_str
    output += ")"
    return output


def main():

#    lang = sys.argv[1]
#    asr = sys.argv[2]
#    ger = sys.argv[3]
#    setting = sys.argv[4]

#    lang = "en"
#    ger = "llama3"
#    ger = "whisper"
#    asr = "small"

    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    lang = sys.argv[3]
#    metric = sys.argv[4]

#    if metric == "ter":
    results = calculate_ter(ref_file, hyp_file, lang)
    print(hyp_file, results)
#    elif metric == "bleu":
#        results = calculate_bleu()


if __name__ == "__main__":
    main()

