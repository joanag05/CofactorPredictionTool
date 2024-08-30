import time
from typing import Callable, Union
from Bio import Entrez
from urllib3 import HTTPResponse


class NCBI:
    def __init__(self, user_entrez):
        self.user_entrez = user_entrez

    def entrez_record(self, callback: Callable, parsing_mode: Union[Callable, None], attempt_number: int = 0, **kwargs) -> Union[HTTPResponse, dict, None]:
        if attempt_number > 10:
            print(str(callback.__name__) + ' unsuccessful!')
            return None
        try:
            handle = callback(**kwargs)
            if attempt_number > 0:
                print("Success at try %d!" % attempt_number)
            if parsing_mode:
                return parsing_mode(handle)
            else:
                return handle
        except Exception as e:
            if 'Invalid uid' in str(e) and 'id' in kwargs:
                invalid_id = str(e).split(" ")[2]
                kwargs['id'] = kwargs['id'].replace(invalid_id, "")
                return self.entrez_record(callback, parsing_mode, attempt_number, **kwargs)
            print('\nError occurred during ' + str(callback.__name__) + "!\nTrying again...")
            print(e)
            time.sleep(5)
            attempt_number += 1
            return self.entrez_record(callback, parsing_mode, attempt_number, **kwargs)


    def entrez_fetch(self, db: str, identifiers: str, rettype: str = "fasta", retmode: str = "xml", **kwargs) -> HTTPResponse:
        try:
            handle = self.entrez_record(self.user_entrez.efetch, parsing_mode=None, db=db, id=identifiers, rettype=rettype, retmode=retmode, max_tries=10, **kwargs)
            return handle
        except Exception as e:
            print(e)

    def entrez_post(self, db: str, **kwargs) -> HTTPResponse:
        try:
            handle = self.entrez_record(self.user_entrez.epost, parsing_mode=None, db=db, attempt_number=9, **kwargs)
            return handle
        except Exception as e:
            print(e)

