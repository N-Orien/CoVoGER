from bert_score import score
import argparse

def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

def main(hyp_file, ref_file, lang):
    hyp_lines = read_lines(hyp_file)
    ref_lines = read_lines(ref_file)

    # Calculate BERTScore
    P, R, F1 = score(hyp_lines, ref_lines, lang=lang)
               
    # Print and return average scores
#    print("Average Precision:", P.mean().item())
#    print("Average Recall:", R.mean().item())
    print("Average F1 Score:", F1.mean().item())
    return P.mean().item(), R.mean().item(), F1.mean().item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BLEURT scores for hypothesis and reference files.")
    parser.add_argument("--hyp", type=str, help="Path to the hypothesis file")
    parser.add_argument("--ref", type=str, help="Path to the reference file")
    parser.add_argument("--lang", type=str, help="Language code")
    args = parser.parse_args()
    
    main(args.hyp, args.ref, args.lang)
