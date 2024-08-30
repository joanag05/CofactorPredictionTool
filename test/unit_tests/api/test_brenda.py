import sys
import unittest

sys.path.append(r'C:\Users\joana\OneDrive - Universidade do Minho\Documentos\GitHub\Cofactor_Prediction_Tool\src')
from cofactor_prediction_tool.api.brenda import BrendaAPI


class TestBrenda(unittest.TestCase):

    def setUp(self):
        import os
        print(os.getcwd())
        self.datafile = r'C:\Users\joana\OneDrive - Universidade do Minho\Documentos\GitHub\Cofactor_Prediction_Tool\data\brenda_download.txt'
        self.brenda = BrendaAPI(self.datafile, 'test/data/outputbrenda.tsv')

    def test_query_all_cofactor_types(self):
        self.brenda.parse_and_create_dataframe()

if __name__ == '__main__':
    unittest.main()
