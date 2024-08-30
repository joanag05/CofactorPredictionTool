import torch
import esm
import pandas as pd
from tqdm import tqdm
from pathlib import Path

class EmbeddingsESM2:
    """
    Class for computing ESM-2 embeddings for proteins.

    Args:
        folder_path (str): The path to the folder where the embeddings will be saved.
        data_path (str): The path to the data file containing RBP sequences.

    Attributes:
        folder_path (str): The path to the folder where the embeddings will be saved.
        data_path (str): The path to the data file containing RBP sequences.

    Methods:
        fetch_dpo_sequences: Fetches the RBP sequences from the data file.
        compute_esm2_embeddings: Computes the ESM-2 embeddings for the RBP sequences and saves them to a file.
    """

    def __init__(self, folder_path, data_path):
        self.folder_path = folder_path
        self.data_path = data_path

    def load_data(self):
        """
        Fetches the RBP sequences from the data file.

        Returns:
            pandas.DataFrame: A DataFrame containing the RBP sequences.
        """
        return pd.read_csv(self.data_path, sep='\t', index_col=0)

    def compute_esm2_embeddings(self):
        """
        Computes the ESM-2 embeddings for the RBP sequences and saves them to a file.
        """
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval() 
        RBPbase = self.load_data()
        print(RBPbase)
        bar = tqdm(total=len(RBPbase['Sequence']), position=0, leave=True)
        sequence_representations = []
        for i, sequence in enumerate(RBPbase['Sequence']):
            data = [(RBPbase.index[i], sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            for j, (_, seq) in enumerate(data):
                sequence_representations.append(token_representations[j, 1: len(seq) + 1].mean(0))
            bar.update(1)
        bar.close()
        sequence_representations_df = pd.DataFrame(sequence_representations).astype('float')
        sequence_representations_df.columns = [f'embedding_{i}' for i in range(sequence_representations_df.shape[1])]
        sequence_representations_df.index = RBPbase.index
        sequence_representations_df.to_csv(self.folder_path, sep='\t', index=True)

