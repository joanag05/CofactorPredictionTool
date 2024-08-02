import pandas as pd
import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")
from cofactor_prediction_tool.preprocessing.EmbeddingsProtFlash import EmbeddingsProtFlash
import time
import os

DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/Final/'
os.chdir(DATA_PATH)

def main(input_paths, output_paths):
    for input_path, output_path in zip(input_paths, output_paths):
        output_file_name = input_path.split('/')[-1].replace('.tsv', '_protflash_embeddings.tsv')
        embeddings = EmbeddingsProtFlash(output_path, input_path)
        embeddings.compute_protflash_embeddings(output_file_name=output_file_name)


if __name__ == "__main__":
    init_time = time.time()
    input_paths = ['final_dataset.tsv']
    output_paths = [DATA_PATH for _ in range(len(input_paths))]
    main(input_paths, output_paths)
    print("Execution time:", time.time() - init_time)