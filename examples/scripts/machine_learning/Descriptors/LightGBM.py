import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")

from cofactor_prediction_tool.machine_learning.MLEmbeddings import ML
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
import os
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier


DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/new_datasets/descriptors/'
os.chdir(DATA_PATH)

dataset_filenames = ["univariate_descriptors1.tsv", "univariate_descriptors2.tsv", 
                     "univariate_descriptors3.tsv", "meta_descriptors1.tsv", "meta_descriptors2.tsv","meta_descriptors3.tsv"]

# Prepend the directory path to each filename
dataset_paths = [os.path.join(DATA_PATH, filename) for filename in dataset_filenames]

output_dir = '/home/jgoncalves/cofactor_prediction_tool/data/new_datasets/descriptors/'

classifier = MultiOutputClassifier(LGBMClassifier(n_estimators=1000, random_state=42))
ml = ML(dataset_paths, "Label Powerset (Random Forest)", classifier)
ml.classify_and_save_results(output_dir)