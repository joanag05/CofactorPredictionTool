import pandas as pd
from propythia.protein.sequence import ReadSequence
from propythia.protein.descriptors import ProteinDescritors

class ProteinCofactorsDescriptors(ProteinDescritors):
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path, sep='\t', index_col=0)
        self.df_binarized = self.df.copy()
        self.preprocess_data()
        self.read_sequences()
        super().__init__(self.res, col='Sequence')

    def preprocess_data(self):
        cofactors = ['NAD', 'NADP','FAD','SAM','P5P',
                     'CoA','THF','FMN','ThDP',
                     'GSH',	'Ubiquinone','Plastoquinone' ,	
                     'Ferredoxin','Ferricytochrome']
        
        cofactors_to_drop = [cofactor for cofactor in cofactors if cofactor in self.df_binarized.columns]
        self.df_binarized.drop(cofactors_to_drop, axis=1, inplace=True)
        return self.df_binarized
          
    def read_sequences(self):
        read_seqs = ReadSequence()
        self.res = read_seqs.par_preprocessing(dataset= self.df_binarized, col = 'Sequence', B ='N', Z = 'Q', U = 'C', O = 'K', J = 'I', X = '')
        return self.res

    def get_all_descriptors(self):
        self.dataset['ProteinID'] = self.dataset.index
        self.all_descriptors = self.get_all()
        self.all_descriptors.index = self.all_descriptors['ProteinID']
        self.all_descriptors.drop('ProteinID', axis=1, inplace=True)
        self.all_descriptors = self.all_descriptors.merge(self.df.drop(['Sequence'], axis=1), left_index=True, right_index=True, how='inner')
        return self.all_descriptors

    def create_final_dataset(self):       
        overlapping_columns = self.all_descriptors.columns.intersection(self.df.columns).tolist()
        if 'ProteinID' in overlapping_columns:
            overlapping_columns.remove('ProteinID')
        self.df.drop(overlapping_columns, axis=1, inplace=True)
        final_dataset = pd.concat(self.df, self.all_descriptors, axis=1)
        final_dataset = final_dataset.dropna()

        columns_order = ['ProteinID', 'Sequence'] + [col for col in final_dataset.columns if col not in ['ProteinID', 'Sequence']]
        final_dataset = final_dataset.reindex(columns=columns_order)

        return final_dataset