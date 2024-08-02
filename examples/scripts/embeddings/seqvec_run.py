
import pandas as pd
import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")
import os
from cofactor_prediction_tool.seqvec.EmbeddingsSeq2Vec import EmbeddingsSeq2Vec
DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/Final/'
os.chdir(DATA_PATH)

def main(input_paths, output_paths):
    for input_path, output_path in zip(input_paths, output_paths):
        embeddings = EmbeddingsSeq2Vec(output_path, input_path)
        embeddings.compute_seqvec_embeddings()

if __name__ == "__main__":
    print(os.getcwd())
    input_paths = ['final_dataset.tsv']
    output_paths = ['SeqVec_embeddings.tsv']
    main(input_paths, output_paths)