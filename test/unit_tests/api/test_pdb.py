import unittest
import sys
sys.path.append(r'C:\Users\joana\OneDrive - Universidade do Minho\Documentos\GitHub\Cofactor_Prediction_Tool\src')
from cofactor_prediction_tool.api.pdb import PdbAPI


class TestPDB(unittest.TestCase):

    def setUp(self):
        self.pdb = PdbAPI('test/data/outputpdb.tsv')

    def test_pdb_dataframe_generation_and_saving(self):
        self.pdb.search("NAD", "NAD")
        self.pdb.search("NADP", "NAP")
        self.pdb.search("S-Adenosyl-L-methionine", "SAM")
        self.pdb.search("FAD", "FAD")
        self.pdb.get_dataframe()
        self.pdb.save_dataframe('test/data/outputpdb.tsv')

if __name__ == '__main__':
    unittest.main()
    
