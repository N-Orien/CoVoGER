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

# Bootstrap hypothesis testing function
def bootstrap_test(scores1, scores2, num_iterations, seed):
    np.random.seed(seed)
    observed_diff = np.mean(scores1) - np.mean(scores2)
    combined_scores = np.concatenate([scores1, scores2])
                                
    count = 0
    for i in range(num_iterations):
        shuffle_combined = np.random.choice(combined_scores, size=len(combined_scores), replace=True)
        new_scores1 = shuffle_combined[:len(scores1)]
        new_scores2 = shuffle_combined[len(scores1):]
                                                                            
        new_diff = np.mean(new_scores1) - np.mean(new_scores2)
        if np.abs(new_diff) >= np.abs(observed_diff):
            count += 1
        print(f"Finished iterations {i}/{num_iterations}")
                                                                                                                
    p_value = count / num_iterations
    return p_value, observed_diff

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate BLEURT scores and perform bootstrap hypothesis testing.')
    parser.add_argument('--ref', type=str, required=True, help='Path to the file containing reference sentences.')
    parser.add_argument('--hyp1', type=str, required=True, help='Path to the file containing hypothesis set 1 sentences.')
    parser.add_argument('--hyp2', type=str, required=True, help='Path to the file containing hypothesis set 2 sentences.')
    parser.add_argument('--checkpoint', type=str, default="/mnt/zamia/zd-yang/works/bleurt_sig/BLEURT-20", help='Path to the BLEURT checkpoint directory.')
    parser.add_argument('--iteration', type=int, default=1000, help='Number of iterations for the bootstrap test.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the bootstrap test.')
                            
    args = parser.parse_args()

    # Read files
    references = read_lines(args.ref)
    hypothesis1 = read_lines(args.hyp1)
    hypothesis2 = read_lines(args.hyp2)

    # Compute BLEURT scores
    bleurt_scores_hyp1 = compute_bleurt_scores(args.checkpoint, references, hypothesis1)
    bleurt_scores_hyp2 = compute_bleurt_scores(args.checkpoint, references, hypothesis2)

    
    # Conduct bootstrap hypothesis test
    p_value, observed_diff = bootstrap_test(np.array(bleurt_scores_hyp1), np.array(bleurt_scores_hyp2), args.iteration, args.seed)


    print("BLEURT Scores for Hypothesis 1:", bleurt_scores_hyp1)
    print("BLEURT Scores for Hypothesis 2:", bleurt_scores_hyp2)
    print(f"Observed difference in means: {observed_diff}")
    print(f"P-value: {p_value}")

    with open(args.hyp1 + ".bleurt", 'w') as file:
        for score in bleurt_scores_hyp1:
            file.write(str(score) + '\n')

    # Save BLEURT scores for Hypothesis 2
    with open(args.hyp2 + ".bleurt" , 'w') as file:
        for score in bleurt_scores_hyp2:
            file.write(str(score) + '\n')
    if p_value < 0.05:
        print("The difference in BLEURT scores between Hypothesis 1 and Hypothesis 2 is statistically significant.")
    else:
        print("No significant difference in BLEURT scores was found.")
