from sacrebleu import corpus_bleu
import scipy.stats as stats
import numpy as np

# Function to read file content
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Read the translations and the reference
#hyp_a = read_file('./st1922_dual_sp_sl/exp/1922_sp_spmt_1/decode_1922_sp_spmt_1_decode_pytorch_transformer_stb5.ja-en_test_ja-en_model.val5.avg.bleu.best/hyp.wrd.trn.detok.en')
#hyp_a = read_file('./st1922_dual_sp_sl/exp/1922_sp_noia_1/decode_1922_sp_noia_1_decode_pytorch_transformer_stb5.ja-en_test_ja-en_model.val5.avg.bleu.best/hyp.wrd.trn.detok.en')
#hyp_a = read_file('./st1922_dual_sp_sl/exp/1922_e2e_1/decode_1922_e2e_1_decode_pytorch_transformer_stb5.ja-en_test_ja-en_model.val5.avg.best.25/hyp.wrd.trn.detok.en')
hyp_a = read_file('./results/gentrans_bmeld_en_zh_st_large.txt.tk')
hyp_b = read_file('./results/gentrans-pemoc_bmeld_en_zh_st_large.txt.tk')
ref = read_file('./results/reference.bmeld.zh.tk')  # Assuming single reference; adjust if multiple

# Calculate BLEU scores for each sentence
# Note: In practice, you might want to calculate corpus-level scores or modify this approach depending on your exact requirements
scores_a = np.array([corpus_bleu([hyp], [[ref_line]]).score for hyp, ref_line in zip(hyp_a, ref)])
scores_b = np.array([corpus_bleu([hyp], [[ref_line]]).score for hyp, ref_line in zip(hyp_b, ref)])
print("calculated")

# Compute the observed difference in mean scores
observed_diff = np.mean(scores_a - scores_b)

# Bootstrap resampling
n_iterations = 1000
n_size = len(scores_a)
diffs = []

for i in range(n_iterations):

    print(i)
    # Resample with replacement
    indices = np.random.choice(range(n_size), n_size, replace=True)
    resampled_a = scores_a[indices]
    resampled_b = scores_b[indices]
                        
    # Recalculate difference for the resampled data
    diff = np.mean(resampled_a - resampled_b)
    diffs.append(diff)

# Calculate the p-value
p_value = np.sum(np.abs(diffs) >= np.abs(observed_diff)) / n_iterations

print(f"Observed difference: {observed_diff}")
print(f"P-value: {p_value}")
