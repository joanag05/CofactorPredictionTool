from rcsbsearchapi.search import AttributeQuery
import pandas as pd
import urllib3
from tqdm import tqdm

class PdbAPI:
    def __init__(self, output_file):
        self.cofactors = {}
        self.output_file = output_file
        self.df = None
        self.seq = {}

    def search(self, cofactor_name, value):
        query = AttributeQuery(attribute="rcsb_nonpolymer_instance_annotation.comp_id", operator="exact_match", value=value).and_(AttributeQuery(attribute="rcsb_nonpolymer_instance_annotation.type", operator="exact_match", value="HAS_NO_COVALENT_LINKAGE"))
        results = set(query())
        for result in results:
            if result not in self.cofactors:
                self.cofactors[result] = set()
            self.cofactors[result].add(cofactor_name)
        self.seq.update(download_pdb_batch(list(self.cofactors.keys())))
            
    def get_dataframe(self):
        self.df = pd.DataFrame(self.cofactors.items(), columns=['ProteinID', 'Cofactor'])
        self.df = pd.get_dummies(self.df.explode('Cofactor'), columns=['Cofactor'], prefix='', prefix_sep='')
        self.df = self.df.groupby('ProteinID').sum()
        self.df = self.df.astype(bool).astype(int)
        self.df['Sequence'] = self.df.index.map(self.seq)
        

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
            self.df.to_csv(file_path, sep='\t', index=True)  
        else:
            raise ValueError("DataFrame is empty.")

def download_pdb(pdb_name):
        """
        Download a pdb file from the PDB. Need an internet connection.
    
        Parameters
        ----------
        Name of the pdb file. (ex: 1dpx or 1dpx.pdb)
    
        Return
        ------
        The pdb file
        """
        # URL of the PDB
        url = 'https://www.rcsb.org/fasta/entry/'
        http = urllib3.PoolManager()
        url = url + pdb_name
        try:
            req = http.request('GET', url, preload_content = False)
        except urllib3.exceptions.MaxRetryError:
            print('Error: Max retries exceeded')
            return None
        except urllib3.exceptions.NewConnectionError:
            print('Error: Could not establish a new connection')
            return None
        except urllib3.exceptions.ConnectionError:
            print('Error: Connection error')
            return None
        except urllib3.exceptions.TimeoutError:
            print('Error: Timeout error')
            return None
        except Exception as e:
            print('Error: ', e)
            return None
        seq = ''
        while True:
            data = req.read()
            if not data:
                break
            seq += data.decode('utf-8')
        req.release_conn()
        return seq

def download_pdb_batch(id_list):

    batch_size = 200
    count = len(id_list)
    gene_id_seq_map = {}
    for start in tqdm(range(0, count, batch_size), colour='green'):
        gene_identifiers = ','.join(e for e in id_list[start:start + batch_size])

        result = download_pdb(gene_identifiers).split('>')
        for i in range(1, len(result)):
            gene_id_seq_map[result[i].split('|')[0].split('_')[0]] = result[i].split('\n')[1]
    return gene_id_seq_map


        
