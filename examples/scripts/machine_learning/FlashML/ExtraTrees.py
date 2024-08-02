import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")

from cofactor_prediction_tool.machine_learning.MLEmbeddings import ML
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
import os



DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/'
os.chdir(DATA_PATH)

dataset_filenames = ['final_dataset_protflash_embeddings.tsv']

dataset_paths = [os.path.join(DATA_PATH, filename) for filename in dataset_filenames]

output_dir = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/'

classifier = MultiOutputClassifier(ExtraTreesClassifier(random_state=42))

ml = ML(dataset_paths, "Extra Trees", classifier)
ml.classify_and_save_results(output_dir)