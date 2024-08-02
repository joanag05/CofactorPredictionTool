import pandas as pd
from propythia.feature_selection import FeatureSelection as PropythiaFeatureSelection
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

class FeatureSelection:
    def __init__(self, input_file1, input_file2, output_file1, output_file2, output_file3):
        self.input_file1 = input_file1
        self.input_file2 = input_file2
        self.output_file1 = output_file1
        self.output_file2 = output_file2
        self.output_file3 = output_file3
        self.df_descriptores = pd.read_csv(self.input_file1, sep='\t')
        df_cofactores = pd.read_csv(self.input_file2, sep='\t')
        self.df_descriptores = pd.concat([self.df_descriptores, df_cofactores], axis=1).drop(columns=['Sequence'])
        self.cofactors = ['NAD', 'NADP', 'FAD', 'SAM']

    def perform_feature_selection(self, df, cofactor, score_func, mode=None, param=None, model=None):
        x_original = df.drop(['ProteinID', cofactor], axis=1)
        target = df[cofactor]
        columns_names = x_original.columns.tolist()

        feature_selection = PropythiaFeatureSelection(x_original=x_original.values, target=target.values, columns_names=columns_names)
        if model:
            _, _, column_selected, _, _ = feature_selection.run_from_model(model=model)
        else:
            _, _, _, column_selected, _, _ = feature_selection.run_univariate(score_func=score_func, mode=mode, param=param)

        df_selected = df.iloc[:, column_selected].copy()
        df_selected[cofactor] = target

        return df_selected

    def process_files(self):
        df_all_cofactors = pd.DataFrame(index=self.df_descriptores.index)

        for cofactor in self.cofactors:
            df_selected = self.perform_feature_selection(self.df_descriptores, cofactor, mutual_info_classif, mode='percentile', param=50, model=SVC(kernel="linear"))
            df_all_cofactors = pd.concat([df_all_cofactors, df_selected], axis=1)
        df_all_cofactors = df_all_cofactors.loc[:,~df_all_cofactors.columns.duplicated()]

        self.save_files(df_all_cofactors, self.output_file1) 

    def save_files(self, df, output_file):
        df.to_csv(output_file, sep='\t', index=False)