import unittest
import pandas as pd
import sys
sys.path.append(r'C:\Users\joana\OneDrive - Universidade do Minho\Documentos\GitHub\Cofactor_Prediction_Tool\src')
from cofactor_prediction_tool.feature_selection.protein_descriptors import ProteinDescriptors

class TestProteinDescriptors(unittest.TestCase):
    def setUp(self):
        self.data_path = 'test/data/cofactors_preprocessed_data.tsv'
        self.protein_descriptors = ProteinDescriptors(self.data_path)

   
if __name__ == '__main__':
    unittest.main()