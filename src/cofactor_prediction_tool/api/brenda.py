
import re
import pandas as pd
from brendapyrser import BRENDA


class BrendaAPI:
    """
    A class used to interact with the BRENDA database.

    Attributes
    ----------
    brenda : BRENDA
        The BRENDA object used to parse the data.
    df : pandas.DataFrame
        The DataFrame containing the parsed data.
    output_file : str
        The path to the file where the DataFrame will be saved.

    Methods
    -------
    parse_and_create_dataframe()
        Parses the BRENDA data, creates a DataFrame, and filters it to keep only the cofactors of interest.
    save_dataframe(file_path)
        Saves the DataFrame to a file.
    """


    def __init__(self, data_file, output_file):
        """
        Constructs all the necessary attributes for the BrendaAPI object.

        Parameters
        ----------
            data_file : str
                The path to the BRENDA data file.
            output_file : str
                The path to the file where the DataFrame will be saved.

Raises

        ValueError
        If no data is available to create the DataFrame or if the DataFrame is empty.

        """
        self.brenda = BRENDA(data_file)
        self.df = None
        self.output_file = output_file

    def parse_and_create_dataframe(self):
        """
        Parses the BRENDA data, creates a DataFrame, and filters it to keep only the cofactors of interest.

        The method iterates over the reactions in the BRENDA data, extracts the protein and cofactor information,
        and creates a DataFrame with this information. Then, it filters the DataFrame to keep only the columns
        corresponding to the cofactors of interest.

        Raises
        ------
        ValueError
            If no data is available to create the DataFrame or if the DataFrame is empty.
        """
        protein_cofactor = {}
        cofactors_to_check = ['ProteinID', 'Organism', 'NADH','NAD', 'NAD+', 'FAD', 'FADH2', 'NADP+', 'NADPH', 'S-adenosyl-L-methionine']
        
        for reaction in self.brenda.reactions:
            for cofactor, data in reaction.cofactors.items():
                # Only continue if cofactor is in the list ['ProteinID', 'Organism', 'NAD', 'NAD+', 'FAD', 'FADH2', 'NADP+', 'NADPH', 'S-adenosyl-L-methionine']
                if cofactor not in cofactors_to_check:
                    continue

                protein_ids = set(re.findall(r'#(\d+)#', data['meta']))
                for protein in protein_ids:
                    temp_protein = reaction.proteins[protein]
                    if temp_protein['proteinID']:
                        protein_id = temp_protein['proteinID'].split()[0].replace('UniProt', '').replace('Swiss-Prot', '')
                        if protein_id not in protein_cofactor:
                            protein_cofactor[protein_id] = (temp_protein['name'], {cofactor})
                        else:
                            protein_cofactor[protein_id][1].add(cofactor)

        if protein_cofactor:
            df = pd.DataFrame(protein_cofactor).T
            df = df.reset_index()
            df.columns = ['UniprotID', 'Organism', 'Cofactors']
            df.rename(columns={'UniprotID': 'ProteinID'}, inplace=True)
            unique_cofactors = set(cofactor for cofactor_list in df['Cofactors'] for cofactor in cofactor_list)
            for cofactor in unique_cofactors:
                df[cofactor] = df['Cofactors'].apply(lambda x: 1 if cofactor in x else 0)
            df.drop(columns=['Cofactors'], inplace=True)
            self.df = df[cofactors_to_check]
        else:
            raise ValueError("No data to create DataFrame.")

        self.save_dataframe(self.output_file)


    def save_dataframe(self, file_path):
        """
        Saves the DataFrame to a file.

        Parameters
        ----------
        file_path : str
            The path to the file where the DataFrame will be saved.

        Raises
        ------
        ValueError
            If the DataFrame is empty.
        """
        if self.df is not None:
            self.df.to_csv(file_path, sep='\t', index=False)
        else:
            raise ValueError("DataFrame is empty.")





