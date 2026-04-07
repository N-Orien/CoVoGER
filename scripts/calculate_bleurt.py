from bleurt import score
import argparse

def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

def main(hyp_file, ref_file, bleurt_checkpoint):
    hyp_lines = read_lines(hyp_file)
    ref_lines = read_lines(ref_file)

    scorer = score.BleurtScorer(bleurt_checkpoint)
    scores = scorer.score(references=ref_lines, candidates=hyp_lines)

    avg_score = sum(scores) / len(scores)
    print("Average BLEURT Score:", avg_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BLEURT scores for hypothesis and reference files.")
    parser.add_argument("--hyp", type=str, help="Path to the hypothesis file")
    parser.add_argument("--ref", type=str, help="Path to the reference file")
    args = parser.parse_args()

    bleurt_checkpoint = "/mnt/zamia/zd-yang/works/bleurt_sig/BLEURT-20"  # E.g., "bleurt-base-128"
    
    main(args.hyp, args.ref, bleurt_checkpoint)
