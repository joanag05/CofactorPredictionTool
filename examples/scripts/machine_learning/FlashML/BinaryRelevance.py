import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")

from cofactor_prediction_tool.machine_learning.MLEmbeddings import ML
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
import os

DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/nFinal/Flash/'
os.chdir(DATA_PATH)

dataset_filenames = ['final_dataset_protflash_embeddings.tsv']

dataset_paths = [os.path.join(DATA_PATH, filename) for filename in dataset_filenames]

output_dir = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/'

classifier = BinaryRelevance(classifier=RandomForestClassifier(n_estimators=1000, random_state=42, criterion='entropy'), require_dense=[True, True])

ml = ML(dataset_paths, "Binary Relevance (Random Forest)", classifier)
ml.classify_and_save_results(output_dir)