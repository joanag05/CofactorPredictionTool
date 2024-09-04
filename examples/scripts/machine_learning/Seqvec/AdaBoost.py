import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")

from cofactor_prediction_tool.machine_learning.MLEmbeddings import ML
from sklearn.ensemble import AdaBoostClassifier
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
            'estimator__learning_rate': [0.1, 0.5, 1.0],
            'estimator__algorithm': ['SAMME', 'SAMME.R']
        }

classifier = MultiOutputClassifier(AdaBoostClassifier(random_state=42))

ml = ML(dataset_paths, "AdaBoost", classifier,param_grid=param_grid)
ml.classify_and_save_results(output_dir)