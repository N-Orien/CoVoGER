import torch
import sys

def simplify_pt_file(input_path, output_path):
    # Load the original .pt file
    model_data = torch.load(input_path, map_location="cpu")

    # Check if "model" key exists in the loaded dictionary
    if "model" in model_data:
        # Extract the "model" dictionary
        model_dict = model_data["model"]

        # Remove "module.model." prefix from each parameter name
        modified_model_dict = {}
        for key, value in model_dict.items():
            new_key = key.replace("module.model.", "")  # Remove the prefix
            modified_model_dict[new_key] = value

        # Save the modified model dictionary to a new .pt file
        torch.save(modified_model_dict, output_path)
        print(f"Modified .pt file saved as: {output_path}")
    else:
        print("Error: 'model' key not found in the .pt file.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simplify_pt_file.py input_path output_path")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        simplify_pt_file(input_path, output_path)
