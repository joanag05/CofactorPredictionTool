import pandas as pd
import numpy as np
import ast

import pandas as pd
import numpy as np
import ast

class OneHotEncoderProcessor:
    def __init__(self, encoding_path, cofactores_path):
        self.encoding_path = encoding_path
        self.cofactores_path = cofactores_path
        self.df_encoding = None
        self.df_cofactores = None
        self.one_hot_encoding_df = None

    def load_dataframes(self):
        self.df_encoding = pd.read_csv(self.encoding_path, sep='\t')  # Assuming TSV format
        self.df_cofactores = pd.read_csv(self.cofactores_path, sep='\t')  # Assuming TSV format

    def concatenate_dataframes(self):
        self.df_encoding = pd.concat([self.df_encoding, self.df_cofactores], axis=1)
        self.df_encoding = self.df_encoding.loc[:, ~self.df_encoding.columns.duplicated()]

    def process_one_hot_encoding(self):
        one_hot_encoding = self.df_encoding['One_hot_encoding'].apply(lambda x: np.array(ast.literal_eval(x)))
        X = np.array(one_hot_encoding.tolist())
        res = [np.ravel(i) for i in X]
        self.one_hot_encoding_df = pd.DataFrame(res)

    def concatenate_cofactor_columns(self):
        self.one_hot_encoding_df = pd.concat([self.one_hot_encoding_df, self.df_encoding[['NAD', 'NADP', 'FAD', 'SAM']]], axis=1)
        self.one_hot_encoding_df = self.one_hot_encoding_df.loc[:, ~self.one_hot_encoding_df.columns.duplicated()]

    def save_to_tsv(self, filename='one_hot_encoding.tsv'):
        self.one_hot_encoding_df.to_csv(filename, sep='\t', index=False)

    def process_and_save(self, filename='one_hot_encoding.tsv'):
        self.load_dataframes()  # Ensure dataframes are loaded before processing
        self.concatenate_dataframes()
        self.process_one_hot_encoding()
        self.concatenate_cofactor_columns()
        self.save_to_tsv(filename)

processor = OneHotEncoderProcessor('/home/jgoncalves/cofactor_prediction_tool/src/cofactor_prediction_tool/preprocessing/protein_encoding.tsv', 'cofactors_preprocessed_data.tsv')
processor.process_and_save('one_hot_encoding.tsv')
print('Finished')