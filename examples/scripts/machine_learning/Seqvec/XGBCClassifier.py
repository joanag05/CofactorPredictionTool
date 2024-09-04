import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")

from cofactor_prediction_tool.machine_learning.MLEmbeddings import ML
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import os


DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/ml_dl_data/SeqVec'
os.chdir(DATA_PATH)

dataset_filenames = ["seqvec.tsv"]

# Prepend the directory path to each filename
dataset_paths = [os.path.join(DATA_PATH, filename) for filename in dataset_filenames]

output_dir = '/home/jgoncalves/cofactor_prediction_tool/data/ml_dl_data/SeqVec'

param_grid = {
    'estimator__learning_rate': [0.01, 0.1],
    'estimator__gamma': [0.5, 2],
    'estimator__subsample': [0.6, 1.0],
    'estimator__colsample_bytree': [0.6, 1.0],
    'estimator__max_depth': [3,9],
    "estimator__n_estimators": [100, 500],
    "estimator__random_state":[42]

}

classifier = MultiOutputClassifier(XGBClassifier(n_estimators=1000, random_state=42, use_label_encoder=False, eval_metric='logloss'))

ml = ML(dataset_paths, "XGBoost", classifier,param_grid=param_grid)
ml.classify_and_save_results(output_dir)