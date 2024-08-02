import pandas as pd
from propythia.feature_selection import FeatureSelection as PropythiaFeatureSelection
from sklearn.feature_selection import f_classif
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class UnvariateSelector:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.df_descriptores = pd.read_csv(self.input_file, sep='\t', index_col=0)
        self.cofactors = ['NAD', 'NADP','FAD','SAM','P5P',
                        'CoA','THF','FMN','ThDP',
                        'GSH',	'Ubiquinone','Plastoquinone',	
                        'Ferredoxin','Ferricytochrome']

    def split_data(self, test_size=0.2, val_size=0.15):
        train, test = train_test_split(self.df_descriptores, test_size=test_size, random_state=42)
        train, val = train_test_split(train, test_size=val_size,random_state=42)

        return train, val, test
    
    def perform_feature_selection(self, data, cofactor, score_func, mode=None, param=None):
        data = data.fillna(0)
        x_original = data.loc[:, ~data.columns.isin(['ProteinID'] + self.cofactors)]
        target = data[cofactor]
        columns_names = x_original.columns.tolist()

        feature_selection = PropythiaFeatureSelection(x_original=x_original.values, target=target.values, columns_names=columns_names)
        _, _, _, column_selected, _, _ = feature_selection.run_univariate(score_func=score_func, mode=mode, param=param)

        df_selected = data.iloc[:, column_selected].copy()
        df_selected[cofactor] = target

        return df_selected

    def process_files(self):
        train, val, test = self.split_data()

        df_all_cofactors_f_classif = pd.DataFrame(index=train.index)

        for cofactor in tqdm(self.cofactors, desc="Processing cofactors for unvariate selection"):
            df_selected_f_classif = self.perform_feature_selection(cofactor, f_classif, mode='percentile', param=50).fit()
            df_all_cofactors_f_classif = pd.concat([df_all_cofactors_f_classif, df_selected_f_classif], axis=1)

        df_all_cofactors_f_classif = df_all_cofactors_f_classif.loc[:,~df_all_cofactors_f_classif.columns.duplicated()]
        cols = [col for col in df_all_cofactors_f_classif if col not in self.cofactors] + self.cofactors
        df_all_cofactors_f_classif = df_all_cofactors_f_classif[cols]
        
        self.save_files(df_all_cofactors_f_classif, self.output_file)

    def save_files(self, df, output_file):
        """
        Saves the selected features to an output file.

        Args:
            df (pd.DataFrame): The dataframe containing the selected features and the target variable.
            output_file (str): The path to the output file.
        """
        df.to_csv(output_file, sep='\t')