
import pandas as pd
import requests
from bs4 import BeautifulSoup
import sys

sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")
from cofactor_prediction_tool.api.uniprot import check_reviewed_status

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('/home/jgoncalves/cofactor_prediction_tool/data/ml_dl_data/ESM2/esm.tsv', sep='\t')

    accession_numbers = df.iloc[:, 0].tolist()

    reviewed, unreviewed = check_reviewed_status(accession_numbers)

    # Save the results to two separate files
    reviewed_df = pd.DataFrame(reviewed, columns=["Accession"])
    unreviewed_df = pd.DataFrame(unreviewed, columns=["Accession"])

    reviewed_df.to_csv('/home/jgoncalves/cofactor_prediction_tool/data/reviewed_ids.csv', index=False)
    unreviewed_df.to_csv('/home/jgoncalves/cofactor_prediction_tool/data/unreviewed_ids.csv', index=False)

    print("Reviewed IDs saved to 'reviewed_ids.csv'")
    print("Unreviewed IDs saved to 'unreviewed_ids.csv'")
