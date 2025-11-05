import sys
#from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab

tokenizer = TokenizerJaMecab()
for line in sys.stdin:
    break
#    tokenized_text = tokenizer(line)
#    print(tokenized_text)
