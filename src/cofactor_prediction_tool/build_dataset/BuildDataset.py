import sys
import os
import pandas as pd
import re
from brendapy import BrendaParser
from propythia.protein.sequence import ReadSequence
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")
from cofactor_prediction_tool.api.uniprot import Uniprot
from cofactor_prediction_tool.preprocessing import Preprocessing


class BuilDataset:
    """
    A class for building datasets for cofactor prediction.

    Args:
        data_path (str): The path to the data directory.

    Attributes:
        data_path (str): The path to the data directory.
        uniprotapi (Uniprot): An instance of the Uniprot class.
        rhea_reactions (dict): A dictionary mapping cofactor names to Rhea reaction IDs.
        brenda_api (BrendaParser): An instance of the BrendaParser class.
        cofactores_chebi (dict): A dictionary mapping cofactor names to ChEBI IDs.

    Methods:
        retrieve_cofactors: Retrieves cofactors based on ChEBI IDs and Rhea reaction IDs.
        remove_duplicates_and_short_sequences: Removes duplicate sequences and sequences with lengths outside a specified range.
        read_sequences: Reads and preprocesses sequences.
        get_specific_cofactors: Retrieves specific cofactors for a given EC number and protein accession.
        retrieve_cofactors_and_generate_df: Retrieves cofactors, generates a DataFrame, and performs data preprocessing.
        retrieve_swiss_proteins: Retrieves Swiss-Prot proteins.
        subtract_files: Subtracts one DataFrame from another based on protein IDs.
        merge_datasets: Merges two datasets and fills missing values with zeros.
        generate_final_dataset: Generates the final dataset by combining and preprocessing data.

    """

    def __init__(self, data_path):
        self.data_path = data_path
        os.chdir(data_path)
        self.uniprotapi = Uniprot()

        self.rhea_reactions = {'Menaquinone': [27834,27409,74079], 
                               'GSH':[24424,19057,48620,48888,50708],
                               'FAD': [24004,24052,42800,43448,43456,43464,43780],
                                'FMN': [21620,17181,31599],'Plastoquinone': [30287,22148],'Ubiquinone': [27405,63936],'Ferricytochrome': [24236,43108,45032,51224,77903]}

        
        self.brenda_api = BrendaParser()
        self.cofactores_chebi = {
            'SAM': [59789, 57856], 'NAD': [57540, 57945], 'NADP': [58349, 57783], 'FAD': [58307, 57692],
            'CoA': [57288, 57287], 'THF': [67016, 15636], 'FMN': [57618, 58210],
            'Menaquinone': [16374, 18151], 'GSH': [57925, 58297], 'Ubiquinone': [16389, 17976],
            'Plastoquinone': [17757, 62192], 'Ferredoxin': [33738, 33737], 'Ferricytochrome': [29033, 29034]
        }

    def retrieve_cofactors(self, cofactor_list, cofactores_chebi):
        """
        Retrieves cofactors based on ChEBI IDs and Rhea reaction IDs.

        Args:
            cofactor_list (list): A list of cofactor names.
            cofactores_chebi (dict): A dictionary mapping cofactor names to ChEBI IDs.

        Returns:
            dict: A dictionary containing the retrieved cofactors.

        """
        results = {}
        to_ignore = ['2.1.1.23', '3.6.1.22']
        for cofactor_name in cofactores_chebi.keys():
            r = self.uniprotapi.search_by_chebi(cofactores_chebi[cofactor_name])
            for protein in r:
                EC_numbers = [e.get('value', None) for e in protein.get('proteinDescription', {}).get('recommendedName', {}).get('ecNumbers', [])]
                if not any([e for e in EC_numbers if e in to_ignore]):
                    if protein['primaryAccession'] not in results:
                        results[protein['primaryAccession']] = {k: 0 for k in cofactor_list}
                        results[protein['primaryAccession']]['Sequence'] = protein['sequence']['value']
                        results[protein['primaryAccession']]['EC_number'] = EC_numbers

                    results[protein['primaryAccession']][cofactor_name] = 1
      
        for cofactor,rhea in self.rhea_reactions.items():
            r = self.uniprotapi.search_by_rhea(rhea)
            for protein in r:
                    EC_numbers = [e.get('value', None) for e in protein.get('proteinDescription', {}).get('recommendedName', {}).get('ecNumbers', [])]
                    if not any([e for e in EC_numbers if e in to_ignore]):
                        if protein['primaryAccession'] not in results:
                            results[protein['primaryAccession']] = {k: 0 for k in cofactor_list}
                            results[protein['primaryAccession']]['Sequence'] = protein['sequence']['value']
                            results[protein['primaryAccession']]['EC_number'] = EC_numbers

                        results[protein['primaryAccession']][cofactor] = 1           
        return results
    
    def remove_duplicates_and_short_sequences(self, df, min_sequence_length, max_sequence_length):
        df = df.drop_duplicates(subset='Sequence')
        df = df[df['Sequence'].apply(lambda x: len(x) >= min_sequence_length)]
        df = df[df['Sequence'].apply(lambda x: len(x) <= max_sequence_length)]
        return df

    def read_sequences(self, df):
        read_seqs = ReadSequence()
        res = read_seqs.par_preprocessing(dataset=df, col='Sequence', B='N', Z='Q', U='C', O='K', J='I', X='')
        return res

    def get_specific_cofactors(self, ec, protein_acc, cofactors_list):
        cofactors_list = set([e.lower() for e in cofactors_list])
        result = set()
        try:
            proteins = self.brenda_api.get_proteins(ec)
            for p in proteins.values():
                if p.uniprot == protein_acc and p.SP:
                    for reaction in p.SP:
                        metabolites = re.split(r' \+ | = ', reaction["data"])
                        for met in metabolites:
                            if {met}.issubset(cofactors_list):
                                result.add(met)
        except Exception as e:
            print(e)
        finally:
            return result

    def retrieve_cofactors_and_generate_df(self):
        cofactor_list = list(self.cofactores_chebi.keys())
        results = self.retrieve_cofactors(cofactor_list, self.cofactores_chebi)
        df = pd.DataFrame(results).T
        df = self.remove_duplicates_and_short_sequences(df, min_sequence_length=100, max_sequence_length=1000)
        df = self.read_sequences(df)
        df.index.name = 'ProteinID'

        quinone = {'Quinone': [132124, 24646]}
        results = self.retrieve_cofactors(cofactor_list, quinone)
        df2 = pd.DataFrame(results).T
        df2 = self.remove_duplicates_and_short_sequences(df2, min_sequence_length=100, max_sequence_length=1000)
        df2 = self.read_sequences(df2)
        df2.index.name = 'ProteinID'
        df2_asdict = df2.to_dict(orient='index')

        cofactors_list = ["Ubiquinone", "Menaquinone", "Plastoquinone", "FAD"]
        for accession, data in df2_asdict.items():
            cofactors = set()
            for ecnumber in data['EC_number']:
                cofactors.update(self.get_specific_cofactors(ecnumber, accession, cofactors_list))
            if cofactors:
                for cofactor in cofactors:
                    df2.loc[accession, cofactor.capitalize()] = 1
            else:
                df2 = df2.drop(accession)

        df = df.drop(df2.index, errors='ignore')
        df = pd.concat([df, df2])
        df.drop(columns=['EC_number', 'Quinone'], inplace=True)
        return df

    def retrieve_swiss_proteins(self):
        results = {}
        r = self.uniprotapi.search_swiss()
        for protein in r:
            if protein['primaryAccession'] not in results:
                results[protein['primaryAccession']] = {'Sequence': protein['sequence']['value']}
        df = pd.DataFrame(results).T
        df.index.name = 'ProteinID'
        return df

    def subtract_files(self, file1, file2):
        df1 = pd.read_csv(file1, sep='\t', index_col='ProteinID')
        df2 = pd.read_csv(file2, sep='\t', index_col='ProteinID')
        df = df1[~df1.index.isin(df2.index)]
        return df

    def merge_datasets(self, df, df2, cofactor_columns, sample_size):
        df2_unique = df2[~df2['ProteinID'].isin(df['ProteinID'])]
        df2_unique = df2_unique.sample(n=min(len(df2_unique), sample_size))
        df = pd.concat([df, df2_unique])
        df[cofactor_columns] = df[cofactor_columns].fillna(0).astype(int)
        return df

    def generate_final_dataset(self, sample_size=10100):
        df = self.retrieve_cofactors_and_generate_df()
        df2 = self.retrieve_swiss_proteins()
        df2.to_csv(os.path.join(self.data_path, 'swiss.tsv'), sep='\t')
        df.to_csv(os.path.join(self.data_path, 'cofactors_df.tsv'), sep='\t')

        df_subtracted = self.subtract_files('swiss.tsv', 'cofactors_df.tsv')
        df_subtracted = self.remove_duplicates_and_short_sequences(df_subtracted, 45, 5000)
        df_subtracted = self.read_sequences(df_subtracted)
        df_subtracted.to_csv(os.path.join(self.data_path, 'protein_without_cofactor.tsv'), sep='\t', index=False)

        cofactor_columns = ['SAM', 'NAD', 'NADP', 'FAD', 'GSH', 'CoA', 'THF', 'FMN', 'Menaquinone', 'Ubiquinone', 'Plastoquinone', 'Ferredoxin', 'Ferricytochrome']
        merged_df = self.merge_datasets(df, df_subtracted, cofactor_columns, sample_size)
        merged_df = self.remove_duplicates_and_short_sequences(merged_df, 45, 5000)
        merged_df = self.read_sequences(merged_df)

        final_output_file = os.path.join(self.data_path, 'dataset.tsv')
        merged_df.to_csv(final_output_file, sep='\t')
        print(f'Final dataset saved to {final_output_file}')