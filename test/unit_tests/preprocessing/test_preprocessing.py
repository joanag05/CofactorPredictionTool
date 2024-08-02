import unittest
import sys
sys.path.append(r'C:\Users\joana\OneDrive - Universidade do Minho\Documentos\GitHub\Cofactor_Prediction_Tool\src')
from cofactor_prediction_tool.preprocessing import Preprocessing


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.raw_data_paths = ['data/datasets/dataset.tsv']
        self.cofactor_list = ['NAD', 'NADP','FAD','SAM','P5P',
                        'CoA','THF','FMN','ThDP',
                        'GSH',	'Ubiquinone','Plastoquinone' ,	
                        'Ferredoxin','Ferricytochrome']
        self.preprocessing = Preprocessing(self.raw_data_paths, self.cofactor_list)

    #def test_preprocess(self):
        #preprocessed_data = self.preprocessing.preprocess()
        #self.preprocessing.save_preprocessed_data('test/data/preprocessed_data_.tsv')
    
    #def test_get_seqs(self):
        #self.preprocessing.read_preprocessed_data('test/data/preprocessed_data_.tsv')
        #seqs = self.preprocessing.get_sequences()
        #self.preprocessing.save_preprocessed_data('test/data/preprocessed_data_with_sequences_added.tsv')

    def test_drop_duplicates_sequences(self):
        import os
        print(os.getcwd())
        self.preprocessing.read_preprocessed_data('data/datasets/dataset.tsv')
        self.preprocessing.remove_duplicates_and_short_sequences()
        self.preprocessing.save_preprocessed_data('data/dataset_final.tsv')



if __name__ == '__main__':
    unittest.main()