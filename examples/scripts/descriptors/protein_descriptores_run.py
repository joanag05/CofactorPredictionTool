import pandas as pd
import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")
import os
from cofactor_prediction_tool.preprocessing import ProteinCofactorsDescriptors
DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/final_datasets/'
os.chdir(DATA_PATH)

def main(input_paths, output_paths):
    for input_path, output_path in zip(input_paths, output_paths):
        protein_descriptors = ProteinCofactorsDescriptors(input_path)
        all_descriptores = protein_descriptors.get_all_descriptors()
        print("All descriptors obtained")
        all_descriptores.to_csv(output_path, sep='\t')
        print(f"Descriptors saved to {output_path}")

if __name__ == "__main__":
    print(os.getcwd())
    input_paths = ['df.tsv']
    output_paths = ['protein_descriptors.tsv']
    main(input_paths, output_paths)


