import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")

from cofactor_prediction_tool.machine_learning.MLEmbeddings import ML
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/'
import os
os.chdir(DATA_PATH)

dataset_filenames = ['final_dataset_protflash_embeddings.tsv']

dataset_paths = [os.path.join(DATA_PATH, filename) for filename in dataset_filenames]

output_dir = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/'

classifier = MultiOutputClassifier(XGBClassifier(n_estimators=1000, random_state=42, use_label_encoder=False, eval_metric='logloss'))

ml = ML(dataset_paths, "XGBoost", classifier)
ml.classify_and_save_results(output_dir)