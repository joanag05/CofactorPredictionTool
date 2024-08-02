import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")

from cofactor_prediction_tool.machine_learning.MLEmbeddings import ML
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import ClassifierChain
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

DATA_PATH = '/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2'
os.chdir(DATA_PATH)

dataset_filenames = ["esm.tsv"]

# Prepend the directory path to each filename
dataset_paths = [os.path.join(DATA_PATH, filename) for filename in dataset_filenames]

output_dir = '/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2'
parameters = [
           
            {
                'classifier': [RandomForestClassifier()],
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__n_estimators': [20, 30, 10, 50],
                'classifier__random_state': [42],
                'classifier__max_depth': [10, 20, 50],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__bootstrap': [True, False]
            }
        ]

classifier = ClassifierChain(classifier=RandomForestClassifier(n_estimators=1000, random_state=42, criterion='entropy'), require_dense=[True, True])

ml = ML(dataset_paths, "Classifier Chain (Random Forest)", classifier, param_grid=parameters)
ml.classify_and_save_results(output_dir)