import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")
from cofactor_prediction_tool.preprocessing.FeatureSelection import UnvariateSelector
import os
DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/final_datasets/Descriptors'
os.chdir(DATA_PATH)


def main(input_paths, output_paths_univariate):
    for input_path, output_path_univariate in zip(input_paths, output_paths_univariate):
        univariate_selector = UnvariateSelector(input_path, output_path_univariate)
        univariate_selector.process_files()

        print(f"Feature selection completed and saved for {input_path}")

if __name__ == "__main__":
    input_paths = ['descriptors.tsv']
    output_paths_univariate = ['univariate_descriptors.tsv']
    main(input_paths, output_paths_univariate)