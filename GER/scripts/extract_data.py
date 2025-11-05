import json
import argparse
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract and save inferences and ground truths from a JSON file.')
    parser.add_argument('json_file', type=str, help='Path to the input JSON file.')

    # Parse command line arguments
    args = parser.parse_args()
    json_file_path = args.json_file

    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Prepare output paths
    dir_path = os.path.dirname(json_file_path)
    inferences_path = os.path.join(dir_path, 'inferences.txt')
    ground_truths_path = os.path.join(dir_path, 'references.txt')

    # Open output files
    with open(inferences_path, 'w', encoding='utf-8') as inf_file, open(ground_truths_path, 'w', encoding='utf-8') as gt_file:
        for item in data:
            if 'inference' in item and 'ground_truth' in item:
                # Write to inferences file
                inf_file.write(item['inference'] + '\n')
                # Write to ground truths file
                gt_file.write(item['ground_truth'] + '\n')

if __name__ == '__main__':
    main()

