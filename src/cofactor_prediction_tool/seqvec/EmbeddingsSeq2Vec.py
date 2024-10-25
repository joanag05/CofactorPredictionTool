import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from allennlp.commands.elmo import ElmoEmbedder
import os
import time

class EmbeddingsSeq2Vec:
    """
    Class for computing sequence embeddings using SeqVec model.
    Args:
        folder_path (str): The path to the folder where the embeddings will be saved.
        data_path (str): The path to the data file containing sequences.
    Methods:
        fetch_dpo_sequences(): Fetches the DPO sequences from the data file.
        compute_seqvec_embeddings(): Computes the SeqVec embeddings for the DPO sequences and saves them to a file.
    """
        
    def __init__(self, folder_path, data_path, model_dir):
        self.folder_path = folder_path
        self.data_path = data_path
        self.model_dir = model_dir

    def fetch_dpo_sequences(self):
        """
        Fetches DPO sequences from a CSV file.

        Returns:
            pandas.DataFrame: A DataFrame containing the DPO sequences.
        """
        return pd.read_csv(self.data_path, sep='\t')
    
    def compute_seqvec_embeddings(self):
        """
        Computes SeqVec embeddings for protein sequences and saves the embeddings to a CSV file.
        
        Returns:
            None
        """

        # Define local paths for options.json and weights.hdf5
        options_path = Path(self.model_dir) / 'options.json'
        weights_path = Path(self.model_dir) / 'weights.hdf5'

        # Check if the files exist in the local directory
        if not os.path.exists(options_path):
            raise FileNotFoundError(f"{options_path} does not exist. Please check the path.")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"{weights_path} does not exist. Please check the path.")

        # Load the ElmoEmbedder with the local paths
        embedder = ElmoEmbedder(options_path, weights_path, cuda_device=0)

        # Fetch sequences from the data file
        RBPbase = self.fetch_dpo_sequences()
        print(RBPbase)

        # Compute embeddings
        bar = tqdm(total=len(RBPbase['Sequence']), position=0, leave=True)
        sequence_representations = []
        for i, sequence in enumerate(RBPbase['Sequence']):
            data = [(RBPbase['ProteinID'][i], sequence)]
            seqs = [list(seq) for _, seq in data]
            seqs.sort(key=len)
            embedding = embedder.embed_sentences(seqs)
            for j, embbed in enumerate(embedding):
                residue_embd = torch.tensor(embbed).sum(dim=0)
                protein_embd = residue_embd.mean(dim=0)
                sequence_representations.append(protein_embd)
            bar.update(1)
        bar.close()

        # Save the embeddings
        sequence_representations_df = pd.DataFrame(sequence_representations).astype('float')
        sequence_representations_df.columns = [f'embedding_{i}' for i in range(sequence_representations_df.shape[1])]
        embeddings_df = pd.concat([RBPbase, sequence_representations_df], axis=1)
        embeddings_df.to_csv(self.folder_path, sep='\t', index=False)

def main(input_paths, output_paths, model_dir):
    for input_path, output_path in zip(input_paths, output_paths):
        embeddings = EmbeddingsSeq2Vec(output_path, input_path, model_dir)
        embeddings.compute_seqvec_embeddings()

if __name__ == "__main__":
    init_time = time.time()
    print(os.getcwd())
    input_paths = ['/home/jgoncalves/cofactor_prediction_tool/data/dataset/dataset.tsv']
    output_paths = ['SeqVec_embeddings_100.tsv']
    model_dir = '/home/jgoncalves/cofactor_prediction_tool/src/cofactor_prediction_tool/seqvec'  # Set the path where your local files are stored
    main(input_paths, output_paths, model_dir)
    print("Execution time:", time.time() - init_time)
