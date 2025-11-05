import numpy as np
from scipy import stats
from comet import download_model, load_from_checkpoint
import argparse

def read_data(hyp1_path, hyp2_path, ref_path, src_path):
    with open(hyp1_path, 'r', encoding='utf-8') as f:
        hyp1 = [line.strip() for line in f]
    with open(hyp2_path, 'r', encoding='utf-8') as f:
        hyp2 = [line.strip() for line in f]
    with open(ref_path, 'r', encoding='utf-8') as f:
        refs = [line.strip() for line in f]
    with open(src_path, 'r', encoding='utf-8') as f:
        src = [line.strip() for line in f]
    return hyp1, hyp2, refs, src

def sig_test(ours, baseline, ref, src, model):
    ours_data = [{"src": src_line, "mt": hyp_line, "ref": ref_line} for src_line, hyp_line, ref_line in zip(src, ours, ref)]
    baseline_data = [{"src": src_line, "mt": hyp_line, "ref": ref_line} for src_line, hyp_line, ref_line in zip(src, baseline, ref)]
    ours_results = model.predict(ours_data, batch_size=8, progress_bar=True)
    baseline_results = model.predict(baseline_data, batch_size=8, progress_bar=True)
    s, p_value = stats.ttest_rel(ours_results["scores"] , baseline_results["scores"], alternative="greater" )
    ours_score = ours_results["system_score"]
    baseline_score = baseline_results["system_score"]

    return ours_score, baseline_score, s, p_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ours', type=str, help='Our hypothesis file')
    parser.add_argument('--bl', type=str, help='Baseline hypothesis file')
    parser.add_argument('--ref', type=str, help='Reference file')
    parser.add_argument('--src', type=str, help='Source file')
    args = parser.parse_args()

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    
    ours, baseline, ref, src = read_data(args.ours, args.bl, args.ref, args.src)
    ours_score, baseline_score, s, p_value = sig_test(ours, baseline, ref, src, model)
    
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
