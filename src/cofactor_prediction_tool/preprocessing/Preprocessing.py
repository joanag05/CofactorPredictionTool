import pandas as pd
from propythia.protein.sequence import ReadSequence
from Bio import SeqIO
from collections import OrderedDict as OrdererDict

class Preprocessing:
    """
    A class used to preprocess raw data for cofactor prediction.

    Methods:
        read_tsv(input_path): Reads a DataFrame from a TSV file.
        write_tsv(output_path): Writes the DataFrame to a TSV file.
        remove_duplicates_and_short_sequences(): Removes duplicates and short sequences from the DataFrame.
        read_sequences(): Reads sequences from the DataFrame.
    """

    def __init__(self, df=None):
        self.df = df
        self.sequences = OrdererDict({})
    
    def read_fasta(self, fasta_file):
        record = SeqIO.parse(fasta_file, "fasta")
        for i, seq in enumerate(record):
            self.sequences[seq.id] = str(seq.seq)
        self.df = pd.DataFrame.from_dict(self.sequences, orient='index', columns=['Sequence'])

    def read_tsv(self, input_path):
        """
        Reads a DataFrame from a TSV file.

        Args:
            input_path (str): The path to the input file.

        Returns:
            pandas.DataFrame: The read DataFrame.
        """
        self.df = pd.read_csv(input_path, sep='\t')
        return self.df


    def write_tsv(self, output_path):
        """
        Writes the DataFrame to a TSV file.

        Args:
            output_path (str): The path to the output file.
        """
        self.df.to_csv(output_path, sep='\t', index=True)

    def remove_duplicates_and_short_sequences(self):
        """
        Removes duplicates and short sequences from the DataFrame.

        This method removes duplicates from the DataFrame and filters out sequences shorter than 50 amino acids.

        Returns:
            pandas.DataFrame: The processed DataFrame.
        """
        self.df = self.df.drop_duplicates('Sequence', keep='first')
        self.df = self.df[self.df['Sequence'].str.len() < 5000 ]
        self.df = self.df[self.df['Sequence'].str.len() > 45 ]
        return self.df

    def read_sequences(self):
        """
        Reads sequences from the DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with processed sequences.
        """
        read_seqs = ReadSequence()
        self.df = read_seqs.par_preprocessing(dataset=self.df, col='Sequence', B='N', Z='Q', U='C', O='K', J='I', X='')
        return self.df