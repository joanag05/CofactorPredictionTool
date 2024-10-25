import pandas as pd
import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")
import os
import time
from cofactor_prediction_tool.preprocessing.EmbeddingsESM2 import EmbeddingsESM2
DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/experimental'
os.chdir(DATA_PATH)

def main(input_paths, output_paths):
    for input_path, output_path in zip(input_paths, output_paths):
        embeddings = EmbeddingsESM2(output_path, input_path)
        embeddings.compute_esm2_embeddings()
        print(f"Embeddings computed and saved for {input_path}")

if __name__ == "__main__":
    init_time = time.time()
    print(os.getcwd())
    input_paths = ['/home/jgoncalves/cofactor_prediction_tool/data/dataset/dataset.tsv']
    output_paths = ['esm.tsv']
    main(input_paths, output_paths)
    print("Execution time:", time.time() - init_time)