import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")

from cofactor_prediction_tool.machine_learning.MLEmbeddings import ML
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
import os


DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/ml_dl_data/SeqVec'
os.chdir(DATA_PATH)

dataset_filenames = ["seqvec.tsv"]

# Prepend the directory path to each filename
dataset_paths = [os.path.join(DATA_PATH, filename) for filename in dataset_filenames]

output_dir = '/home/jgoncalves/cofactor_prediction_tool/data/ml_dl_data/SeqVec'


param_grid = {
            'estimator__n_estimators': [50, 100, 200],
            'estimator__bootstrap': [True, False],
            'estimator__criterion': ['gini', 'entropy'],
            'estimator__max_depth': [10, 50, 100, None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__max_features': ['auto', 'sqrt', 'log2'],
            'estimator__max_leaf_nodes': [None, 10, 50, 100],
            'estimator__min_impurity_decrease': [0.0, 0.1, 0.2],
            'estimator__random_state': [42],
            
            

        }

classifier = MultiOutputClassifier(ExtraTreesClassifier(random_state=42))

ml = ML(dataset_paths, "Extra Trees", classifier,param_grid=param_grid)
ml.classify_and_save_results(output_dir)