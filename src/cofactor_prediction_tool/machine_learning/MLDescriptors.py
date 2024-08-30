import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix,hamming_loss, jaccard_score,coverage_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from tqdm import tqdm
import os
import numpy as np
from joblib import dump, load
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _validate_shuffle_split
from itertools import chain
from sklearn.model_selection import GridSearchCV

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif, f_classif
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _validate_shuffle_split
from itertools import chain
import os
import numpy as np
from joblib import dump

class MLDescriptors:
    def __init__(self, dataset_paths, classifier_name, classifier, param_grid):
        self.dataset_paths = dataset_paths
        self.classifier = {classifier_name: classifier}
        self.param_grid = param_grid

    def multilabel_train_test_split(self, *arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
        if stratify is None:
            return train_test_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state, stratify=None, shuffle=shuffle)
        
        assert shuffle, "Stratified train/test split is not implemented for shuffle=False"
        
        n_arrays = len(arrays)
        arrays = indexable(*arrays)
        n_samples = _num_samples(arrays[0])
        n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=0.25)
        cv = MultilabelStratifiedShuffleSplit(test_size=n_test, train_size=n_train, random_state=123)
        train, test = next(cv.split(X=arrays[0], y=stratify))

        return list(chain.from_iterable((_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays))

    def apply_feature_selection(self, X_train, y_train, X_test):
        transformer = GenericUnivariateSelect(f_classif, mode='percentile', param=50).fit(X_train, y_train)
        X_train_transformed = transformer.transform(X_train)
        X_test_transformed = transformer.transform(X_test)
        return X_train_transformed, X_test_transformed

    def multilabel_classification_results(self, dataset, name, classifier):
        data = pd.read_csv(dataset, sep='\t', index_col=0)
        data = data.dropna()
        
        cofactors = ['NAD', 'NADP', 'FAD', 'SAM',
                    'CoA', 'THF', 'FMN', 'Menaquinone',
                    'GSH', 'Ubiquinone', 'Plastoquinone',
                    'Ferredoxin', 'Ferricytochrome']
        
        cofactors_in_dataset = [cofactor for cofactor in cofactors if cofactor in data.columns]
        
        X = data.drop(columns=cofactors_in_dataset + ['Sequence'])
        y = data[cofactors_in_dataset]

        X_train, X_test, y_train, y_test = self.multilabel_train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        X_train, X_test = self.apply_feature_selection(X_train, y_train, X_test)
        results = {}
       
        grid_search = GridSearchCV(estimator=classifier, param_grid=self.param_grid, cv=3, n_jobs=-1, verbose=2,scoring='accuracy')
        
        try:
            grid_search.fit(X_train, y_train)
            best_classifier = grid_search.best_estimator_
            dump(best_classifier, "your-model.joblib")
        except Exception as e:
            results[name] = {"error": f"Error during fitting: {str(e)}"}
            print(e)

        try:
            y_pred = best_classifier.predict(X_test)
        except Exception as e:
            results[name] = {"error": f"Error during prediction: {str(e)}"}
            print(e)

        accuracy = accuracy_score(y_test, y_pred)
        hamming = hamming_loss(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        confusion = multilabel_confusion_matrix(y_test, y_pred)
        results[name] = {"accuracy": accuracy, "report": report, "confusion": confusion, "hamming_loss": hamming}

        return results

    def save_results_to_file(self, results, filename):
        with open(filename, 'w') as file:
            for name, result in results.items():
                file.write(f"Classifier: {name}\n")
                file.write("-" * 20 + "\n") 
                if 'error' in result:
                    file.write(f"Error: {result['error']}\n\n")
                else:
                    file.write(f"Accuracy: {result['accuracy']:.2f}\n")
                    file.write(f"Hamming Loss: {result['hamming_loss']:.2f}\n")
                    file.write("Classification Report:\n")
                    file.write(f"{result['report']}\n")
                    file.write("Confusion Matrix:\n")

                    for matrix in result['confusion']:
                        np.savetxt(file, matrix, fmt="%d")
                        file.write("\n")
                    file.write("=" * 40 + "\n\n") 