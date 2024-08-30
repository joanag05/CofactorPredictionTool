
import requests
import pandas as pd
import io
from tqdm import tqdm
import time

class SabioAPI:
    """
    A class used to interact with the SABIO-RK database.

    ...

    Attributes
    ----------
    QUERY_URL : str
        a constant URL used for querying the SABIO-RK database
    output_file : str
        the name of the file where the results will be saved

    Methods
    -------
    get_df_for_cofactor(cofactor, cofactor_type)
        Retrieves cofactor data from the SABIO database based on the given cofactor and cofactor type.
    query_all_cofactor_types()
        Queries all cofactor types from the SABIO database and saves the results in a file.
    """

    QUERY_URL = 'http://sabiork.h-its.org/sabioRestWebServices/kineticlawsExportTsv'

    def __init__(self, output_file):
        """
        Constructs all the necessary attributes for the SabioAPI object.

        Parameters
        ----------
            output_file : str
                the name of the file where the results will be saved
        """
        self.output_file = output_file

    def get_df_for_cofactor(self, cofactor, cofactor_type):
        """
        Retrieves cofactor data from the SABIO database based on the given cofactor and cofactor type.

        Parameters
        ----------
            cofactor : str
                The name of the cofactor.
            cofactor_type : str
                The type of the cofactor.

        Returns
        -------
            pandas.DataFrame
                A DataFrame containing the retrieved cofactor data.
        """
        query_dict = {cofactor_type: cofactor}
        query_string = ' OR '.join(['%s:%s' % (k, v) for k, v in query_dict.items()])
        query = {'fields[]': ['EntryID', 'Organism', 'UniprotID'], 'q': query_string}
        success = False
        while not success:
            try:
                request = requests.post(self.QUERY_URL, params=query)
                request.raise_for_status()
                rawData = pd.read_csv(io.StringIO(request.text), sep='\t')
                rawData.rename(columns={'UniprotID': 'ProteinID'}, inplace=True)  
                success = True
            except requests.exceptions.Timeout as errt:
                time.sleep(10)
            except requests.exceptions.SSLError:
                time.sleep(10)
            except Exception as e:
                print(e)
                success = True
                rawData = pd.DataFrame()
        return rawData

    def query_all_cofactor_types(self):
        """
        Queries all cofactor types from the SABIO database and saves the results in a file.

        The method iterates over a predefined list of cofactors and cofactor types, retrieves the data for each combination
        and concatenates the results in a single DataFrame. The DataFrame is then saved in a file.
        """
        cofactor_list = ['NAD', 'NADH', 'NADP', 'NADPH', 'FAD', 'FADH2', 'S-Adenosyl-L-methionine']
        cofactor_type_list = ['cofactor', 'substrate', 'product']
        main_df = pd.DataFrame()
        pbar = tqdm(total=len(cofactor_list)*len(cofactor_type_list))
        for cofactor in cofactor_list:
            for cofactor_type in cofactor_type_list:
                df = self.get_df_for_cofactor(cofactor, cofactor_type)
                df[cofactor_list] = 0
                df[cofactor] = 1
                df.reset_index(drop=True, inplace=True)
                main_df = pd.concat([main_df, df], ignore_index=True)
                pbar.update(1)
        main_df.to_csv(self.output_file, sep='\t', index=False)
