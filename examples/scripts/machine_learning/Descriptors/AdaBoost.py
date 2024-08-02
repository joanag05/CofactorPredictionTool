import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")

from cofactor_prediction_tool.machine_learning import MLDescriptors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
import os

DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/new_datasets/descriptors/'
os.chdir(DATA_PATH)


dataset_filenames = ['descriptors.tsv']

# Prepend the directory path to each filename
dataset_paths = [os.path.join(DATA_PATH, filename) for filename in dataset_filenames]

output_dir = '/home/jgoncalves/cofactor_prediction_tool/data/new_datasets/descriptors/'
classifier = MultiOutputClassifier(AdaBoostClassifier(random_state=42))

ml = MLDescriptors(dataset_paths, "AdaBoost", classifier)
ml.classify_and_save_results(output_dir)