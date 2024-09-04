import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from allennlp.commands.elmo import ElmoEmbedder
import urllib
import os

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
        
    def __init__(self, folder_path, data_path):
        self.folder_path = folder_path
        self.data_path = data_path

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

        model_dir = Path('./seq2vec')
        Path.mkdir(model_dir, exist_ok=True)
        repo_link = "http://rostlab.org/~deepppi/embedding_repo/embedding_models/seqvec"
        options_link = repo_link + "/options.json"
        weights_link = repo_link + "/weights.hdf5"
        weights_path = model_dir / 'weights.hdf5'
        options_path = model_dir / 'options.json'
        if not os.path.exists(options_path):
            urllib.request.urlretrieve(options_link, str(options_path))
        if not os.path.exists(options_path):
            urllib.request.urlretrieve(weights_link, str(weights_path))
        embedder = ElmoEmbedder(options_path, weights_path, cuda_device=0)
        RBPbase = self.fetch_dpo_sequences()
        print(RBPbase)
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
        sequence_representations_df = pd.DataFrame(sequence_representations).astype('float')
        sequence_representations_df.columns = [f'embedding_{i}' for i in range(sequence_representations_df.shape[1])]
        embeddings_df = pd.concat([RBPbase, sequence_representations_df], axis=1)
        embeddings_df.to_csv(self.folder_path, sep='\t', index=False)