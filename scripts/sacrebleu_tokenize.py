from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
import sys

tokenizer = Tokenizer13a()
for line in sys.stdin:
    tokenized_text = tokenizer(line)
    print(tokenized_text)
