import torch
import esm
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import gc
from ProtFlash.pretrain import load_prot_flash_base
from ProtFlash.utils import batchConverter

class EmbeddingsProtFlash:
    """
    Class for computing ProtFlash embeddings and saving them to a file.

    Args:
        folder_path (str): The path to the folder where the output file will be saved.
        data_path (str): The path to the input data file.

    Methods:
        fetch_dpo_sequences: Fetches the DPO sequences from the input data file.
        compute_protflash_embeddings: Computes ProtFlash embeddings for the DPO sequences and saves them to a file.

    Attributes:
        folder_path (str): The path to the folder where the output file will be saved.
        data_path (str): The path to the input data file.
    """
    
    def __init__(self, folder_path, data_path):
        self.folder_path = folder_path
        self.data_path = data_path

    def fetch_dpo_sequences(self):
        """
        Fetches the DPO sequences from the input data file.

        Returns:
            pandas.DataFrame: A DataFrame containing the DPO sequences.
        """
        return pd.read_csv(self.data_path, sep='\t')

    def compute_protflash_embeddings(self,output_file_name='protflash_embeddings_rbp.tsv'):
        """
        Compute ProtFlash embeddings for the given RNA binding protein (RBP) sequences and save the embeddings to a file.

        Parameters:
        - output_file_name (str): The name of the output file to save the embeddings. Default is 'protflash_embeddings_rbp.tsv'.

        Returns:
        None
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on {device}")
        model = load_prot_flash_base()
        model = model.to(device)
        model.eval()
        RBPbase = self.fetch_dpo_sequences()
        batch_size = 32
        sequence_representations = [] 
        for i in tqdm(range(0, len(RBPbase), batch_size)):
            data = [(i, seq) for i, seq in enumerate(RBPbase['Sequence'][i: i + batch_size])]
            ids, batch_token, lengths = batchConverter(data)
            batch_token = batch_token.to(device)
            with torch.no_grad():
                token_embedding = model(batch_token, lengths)
            for i, (_, seq) in enumerate(data):
                sequence_representations.append(token_embedding[i, 0: len(seq) + 1].mean(0))
            del token_embedding
            gc.collect()
 
        sequence_representations_df = pd.DataFrame(sequence_representations).astype('float')
        sequence_representations_df.columns = [f'embedding_{i}' for i in range(sequence_representations_df.shape[1])]
        embeddings_df = pd.concat([RBPbase, sequence_representations_df], axis=1)
        embeddings_df.to_csv(self.folder_path + '/' + output_file_name, sep='\t', index=False)
        del sequence_representations
        del sequence_representations_df
        del embeddings_df
        gc.collect()