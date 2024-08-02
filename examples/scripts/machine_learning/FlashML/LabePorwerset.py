import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")

from cofactor_prediction_tool.machine_learning.MLEmbeddings import ML
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
import os

DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/'
os.chdir(DATA_PATH)

dataset_filenames = ['final_dataset_protflash_embeddings.tsv']

# Prepend the directory path to each filename
dataset_paths = [os.path.join(DATA_PATH, filename) for filename in dataset_filenames]

output_dir = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/'
classifier = LabelPowerset(classifier=RandomForestClassifier(n_estimators=1000, random_state=42, criterion='entropy'))

ml = ML(dataset_paths, "Label Powerset (Random Forest)", classifier)
ml.classify_and_save_results(output_dir)