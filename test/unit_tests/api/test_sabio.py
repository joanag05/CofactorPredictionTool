import unittest
import sys
sys.path.append(r'C:\Users\joana\OneDrive - Universidade do Minho\Documentos\GitHub\Cofactor_Prediction_Tool\src')
from cofactor_prediction_tool.api.sabio import SabioAPI

class TestSabio(unittest.TestCase):

    def setUp(self):
        self.sabio = SabioAPI('test/data/outputsabio.tsv')

    def test_query_all_cofactor_types(self):
        self.sabio.query_all_cofactor_types()

if __name__ == '__main__':
    unittest.main()