import argparse
import numpy as np
from bleurt import score

def read_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    return lines

# Function to compute BLEURT scores
def compute_bleurt_scores(checkpoint, references, hypotheses):
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=hypotheses)
    print(references[0])
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate BLEURT scores and perform bootstrap hypothesis testing.')
    parser.add_argument('--ref', type=str, required=True, help='Path to the file containing reference sentences.')
    parser.add_argument('--hyp', type=str, required=True, help='Path to the file containing hypothesis set 1 sentences.')
    parser.add_argument('--checkpoint', type=str, default="/mnt/zamia/zd-yang/works/bleurt_sig/BLEURT-20", help='Path to the BLEURT checkpoint directory.')
    parser.add_argument('--iteration', type=int, default=1000, help='Number of iterations for the bootstrap test.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the bootstrap test.')
                            
    args = parser.parse_args()

    # Read files
    references = read_lines(args.ref)
    hypothesis = read_lines(args.hyp)

    # Compute BLEURT score
    bleurt_scores_hyp = compute_bleurt_scores(args.checkpoint, references, hypothesis)

    print("BLEURT Score for Hypothesis:", bleurt_scores_hyp)

    with open(args.hyp + ".bleurt", 'w') as file:
        for score in bleurt_scores_hyp:
            file.write(str(score) + '\n')

