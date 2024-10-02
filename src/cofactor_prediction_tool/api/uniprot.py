import gzip
import json
import re
import shutil
from os import makedirs
from os.path import join, exists
from typing import List
import wget
import requests
from requests.adapters import HTTPAdapter, Retry
import urllib3
import tqdm
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter

import json


def get_next_link(headers):
    """
    Extracts the next link from the headers dictionary.

    Args:
        headers (dict): The headers dictionary containing the "Link" key.

    Returns:
        str: The next link URL if found, None otherwise.
    """
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)


class Uniprot:
    """
    A class for interacting with the UniProt database.
    """

    def __init__(self):
        self.data_dir = join(join("databases", "uniprot"))
        if not exists(self.data_dir):
            makedirs(self.data_dir)
        self.session = None
        self.db_file = join(self.data_dir, "uniprot_sprot.dat")
        self.db_file_compressed = join(self.data_dir, "uniprot_sprot.dat.gz")

    def search(self):
        """
        Search the UniProt database

        Returns:
        -------
        list: A list of search results.
        """
        # TODO: Implement search
        pass

    def get_batch(self, batch_url):
        """
        Retrieves a batch of data from the given batch URL.

        Args:
            batch_url (str): The URL of the batch to retrieve.

        Yields:
            tuple: A tuple containing the response object and the total number of results.

        Raises:
            HTTPError: If the response status code is not successful.
        """

        while batch_url:
            response = self.session.get(batch_url)
            response.raise_for_status()
            total = response.headers["x-total-results"]
            yield response, total
            batch_url = get_next_link(response.headers)



    def search_swiss(self, reviewed: bool = True) -> list:
        """
        Searches for Swiss-Prot entries.

        Args:
            reviewed (bool, optional): Specifies whether to search only reviewed entries. Defaults to True.

        Returns:
        list: A list of Swiss-Prot entries.
        """
        retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        url = f"https://rest.uniprot.org/uniprotkb/search?query=(reviewed:true)&size=500&fields=accession,sequence"
        results = []
        

        batches = list(self.get_batch(url))
        total_batches = len(batches)
        
        for batch, total in tqdm(batches, total=total_batches, desc="Fetching Swiss-Prot entries"):
            data = json.loads(batch.text)
            results += data['results']
        
        return results

    def search_by_chebi(self, chebi: list, reviewed: bool = True) -> list:
        """
        Searches for UniProt entries based on ChEBI IDs.

        Args:
            chebi (list): A list of ChEBI IDs to search for.
            reviewed (bool, optional): Specifies whether to search only reviewed entries. Defaults to True.

        Returns:
        list: A list of UniProt entries matching the search criteria.
        """
        retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        url = f"https://rest.uniprot.org/uniprotkb/search?query=(reviewed:true)%20AND%20(chebi:{str(chebi[0])})%20AND%20(chebi:{str(chebi[1])})&size=500&fields=accession,ec,sequence"
        results = []
        for batch, total in self.get_batch(url):
            data = json.loads(batch.text)
            results += data['results']
        return results
    
    def search_by_rhea(self, rhea: list, reviewed: bool = True) -> list:
            """
            Searches for UniProt entries based on ChEBI IDs.

            Args:
                chebi (list): A list of ChEBI IDs to search for.
                reviewed (bool, optional): Specifies whether to search only reviewed entries. Defaults to True.

            Returns:
            list: A list of UniProt entries matching the search criteria.
            """
            retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
            self.session = requests.Session()
            self.session.mount("https://", HTTPAdapter(max_retries=retries))
            url = f"https://rest.uniprot.org/uniprotkb/search?query=(rhea:{str(rhea[0])})&size=500&fields=accession,ec,sequence"
            results = []
            for batch, total in self.get_batch(url):
                data = json.loads(batch.text)
                results += data['results']
            return results

    
def download_uniprot(uniprot_id):
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
        url = 'https://rest.uniprot.org/uniprotkb/'
        http = urllib3.PoolManager()
        url = url + uniprot_id + '.fasta'
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

def download_uniprot_batch(id_list):
    """
    Downloads the UniProt sequences for a batch of protein IDs.

    Args:
        id_list (list): A list of protein IDs.

    Returns:
        dict: A dictionary mapping protein IDs to their corresponding UniProt sequences.
    """
    count = len(id_list)
    gene_id_seq_map = {}
    for protein_id in tqdm(id_list, colour='green'):
        result = download_uniprot(protein_id).split('>')
        assert len(result) == 1
        gene_id_seq_map[protein_id] = result[0]
    return gene_id_seq_map
        



def check_reviewed_status(accession_numbers):
    """
    Checks if the given accession numbers are reviewed in UniProt.

    Args:
        accession_numbers (list): A list of UniProt accession numbers.

    Returns:
        tuple: Two lists, one with reviewed IDs and another with unreviewed IDs.
    """
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))


    bach_size = 10

    reviewed_ids = []
    unreviewed_ids = []


    for i in range(0, len(accession_numbers), bach_size):
        batch = accession_numbers[i:i + bach_size]
        batch_query = ' OR '.join(batch)
        url = f'https://rest.uniprot.org/uniprotkb/search?query={batch_query}&size=500&fields=accession,reviewed'
        response = session.get(url)
        if response.status_code == 200:
            data = response.json()
            for result in data['results']:
                accession = result['primaryAccession']
                # check if the primary accession is in the list of accessions
                if accession in batch:
                    if 'UniProtKB reviewed (Swiss-Prot)' in result.get("entryType", ""):
                        reviewed_ids.append(accession)
                    else:
                        unreviewed_ids.append(accession)
            

    return reviewed_ids, unreviewed_ids

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('/home/jgoncalves/cofactor_prediction_tool/data/ml_dl_data/ESM2/esm.tsv', sep='\t', usecols=[0])

    accession_numbers = df.iloc[:, 0].tolist()

    reviewed, unreviewed = check_reviewed_status(accession_numbers)

    # Save the results to two separate files
    reviewed_df = pd.DataFrame(reviewed, columns=["Accession"])
    unreviewed_df = pd.DataFrame(unreviewed, columns=["Accession"])
    print(reviewed_df)
    reviewed_df.to_csv('/home/jgoncalves/cofactor_prediction_tool/data/reviewed_ids.csv', index=False)
    unreviewed_df.to_csv('/home/jgoncalves/cofactor_prediction_tool/data/unreviewed_ids.csv', index=False)

    print("Reviewed IDs saved to 'reviewed_ids.csv'")
    print("Unreviewed IDs saved to 'unreviewed_ids.csv'")



