import sys
from jiwer import wer
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh


def calculate_ter(ref_file, hyp_file, lang):
    with open(ref_file, 'r', newline='', encoding='utf-8') as reffile, open(hyp_file, 'r', newline='', encoding='utf-8') as hypfile:
        reflines = reffile.readlines()
        hyplines = hypfile.readlines()

        if lang == "ja":
            tokenizer = TokenizerJaMecab()
        elif lang == "zh":
            tokenizer = TokenizerZh()
        else:
            tokenizer = Tokenizer13a()
        
        refs = [tokenizer(line.strip()) for line in reflines]
        hyps = [tokenizer(line.strip()) for line in hyplines]

        ter = wer(refs, hyps)

        return ter
            
def print_wer(wer_list):
    output = "("
    for wer in wer_list:
        wer_str = f"{wer * 100:.1f}, "
        output += wer_str
    output += ")"
    return output


def main():

    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    lang = sys.argv[3]

    results = calculate_ter(ref_file, hyp_file, lang)
    results = round(results * 100, 1)
#    print(hyp_file, results)
    print(results)


if __name__ == "__main__":
    main()

