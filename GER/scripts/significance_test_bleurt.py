import numpy as np
from scipy import stats
from bleurt import score
import argparse

def read_data(hyp1_path, hyp2_path, ref_path):
    with open(hyp1_path, 'r', encoding='utf-8') as f:
        hyp1 = [line.strip() for line in f]
    with open(hyp2_path, 'r', encoding='utf-8') as f:
        hyp2 = [line.strip() for line in f]
    with open(ref_path, 'r', encoding='utf-8') as f:
        refs = [line.strip() for line in f]
    return hyp1, hyp2, refs

def sig_test(ours, baseline, ref, scorer):
    num_sentences = len(ref)
    ours_scores = scorer.score(references=ref, candidates=ours)
    baseline_scores = scorer.score(references=ref, candidates=baseline)
    s, p_value = stats.ttest_rel(ours_scores , baseline_scores, alternative="greater" )
    ours_score = sum(ours_scores) / num_sentences
    baseline_score = sum(baseline_scores) / num_sentences

    return ours_score, baseline_score, s, p_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ours', type=str, help='Our hypothesis file')
    parser.add_argument('--bl', type=str, help='Baseline hypothesis file')
    parser.add_argument('--ref', type=str, help='Reference file')
    args = parser.parse_args()

    bleurt_checkpoint = "/mnt/zamia/zd-yang/works/bleurt_sig/BLEURT-20"
    scorer = score.BleurtScorer(bleurt_checkpoint)
    
    ours, baseline, ref = read_data(args.ours, args.bl, args.ref)
    ours_score, baseline_score, s, p_value = sig_test(ours, baseline, ref, scorer)
    
    print(f"---=== BLEURT score ===---")
    print(f"Score of Ours: {ours_score:.6f}")
    print(f"Score of Baseline: {baseline_score:.6f}")
    print(f"-----")
    print(f"S: {s}")
    print(f"P-value: {p_value}")
    if p_value < 0.05:
        print("Statistically significant difference found!")
    else:
        print("No significant difference detected.")

if __name__ == "__main__":
    main()
