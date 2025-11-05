import sys
import pandas as pd

def convert_lines(input_file_path):
    csv_data = pd.read_csv(input_file_path, header=None)

    ground_truth_list = []
    for i, row in csv_data.iterrows():
#        if isinstance(row[3], float) or isinstance(row[4], float):
        if isinstance(row[6], float):
            pass
        else:
            if row[6] in ground_truth_list:
                print(row[6])
            else:
                ground_truth_list.append(row[6])
#        if i > 10:
#            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_data.py input_path")
    else:
        input_path = sys.argv[1]
        convert_lines(input_path)
