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

class ML:
    """
    Class for performing multilabel classification using machine learning algorithms.

    Parameters:
    - dataset_paths (list): List of file paths to the datasets.
    - classifier_name (str): Name of the classifier.
    - classifier (object): Machine learning classifier object.
    - param_grid (dict): Dictionary of hyperparameters for grid search.

    Methods:
    - multilabel_train_test_split: Train test split for multilabel classification.
    - classify_and_save_results: Classify the datasets and save the results.
    - multilabel_classification_results: Perform multilabel classification and return the results.
    - save_results_to_file: Save the classification results to a file.
    """

    def __init__(self, dataset_paths, classifier_name, classifier, param_grid):
        self.dataset_paths = dataset_paths
        self.classifier = {classifier_name: classifier}
        self.param_grid = param_grid

    def multilabel_train_test_split(self, *arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
        """
        Train test split for multilabel classification. Uses the algorithm from: 
        'Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data'.
        """
        if stratify is None:
            return train_test_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state, stratify=None, shuffle=shuffle)
        
        assert shuffle, "Stratified train/test split is not implemented for shuffle=False"
        
        n_arrays = len(arrays)
        arrays = indexable(*arrays)
        n_samples = _num_samples(arrays[0])
        n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=0.25)
        cv = MultilabelStratifiedShuffleSplit(test_size=n_test, train_size=n_train, random_state=42)
        train, test = next(cv.split(X=arrays[0], y=stratify))
        X_train, X_test, y_train, y_test = list(chain.from_iterable((_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays))
        #set orinal column names
        X_train.columns = arrays[0].columns
        X_test.columns = arrays[0].columns
        return X_train, X_test, y_train, y_test
    
    def classify_and_save_results(self, output_dir):
        """
        Classify the datasets and save the results.

        Parameters:
        - output_dir (str): Directory to save the results.

        Returns:
        - None
        """
        for dataset_path in self.dataset_paths:
            for classifier_name, classifier in self.classifier.items():
                results, y_test_pred = self.multilabel_classification_results(dataset_path, classifier_name, classifier)
                dataset_name = os.path.basename(dataset_path).split('.')[0]
                output_file_path = os.path.join(output_dir, f"{dataset_name}_{classifier_name}_ML_results.txt")
                self.save_results_to_file(results, output_file_path)
                
                y_test_pred.to_csv(f'y_test_pred_{classifier_name}_{dataset_name}.tsv', sep='\t')

    def multilabel_classification_results(self, dataset, name, classifier):
        """
        Perform multilabel classification and return the results.

        Parameters:
        - dataset (str): File path to the dataset.
        - name (str): Name of the classifier.
        - classifier (object): Machine learning classifier object.

        Returns:
        - results (dict): Dictionary containing the classification results.
        - y_test_pred (DataFrame): DataFrame containing the true and predicted labels.
        """
        data = pd.read_csv(dataset, sep='\t', index_col=0).sample(frac=1).reset_index(drop=True)
        print(data)
        data = data.dropna()
        
        cofactors = ['NAD', 'NADP', 'FAD', 'SAM',
                    'CoA', 'THF', 'FMN', 'Menaquinone',
                    'GSH', 'Ubiquinone', 'Plastoquinone',
                    'Ferredoxin', 'Ferricytochrome']
        
        cofactors_in_dataset = [cofactor for cofactor in cofactors if cofactor in data.columns]
        
        X = data.drop(columns=cofactors_in_dataset)
        y = data[cofactors_in_dataset]

        X_train, X_test, y_train, y_test = self.multilabel_train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        results = {}
        pd.set_option('display.max_columns', None)
        print(y_test)
       
        string_columns = X.select_dtypes(include=['object']).columns


        grid_search = GridSearchCV(estimator=classifier, param_grid=self.param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1_macro')

        grid_search.fit(X_train, y_train)
       
        best_classifier = grid_search.best_estimator_
        dump(best_classifier, f"{name}_model.joblib")

        y_pred = best_classifier.predict(X_test)

        if not isinstance(y_pred, np.ndarray): 
            y_pred_fd = pd.DataFrame.sparse.from_spmatrix(y_pred, columns=cofactors_in_dataset, index=y_test.index)
        else:
            y_pred_fd = pd.DataFrame(y_pred, columns=cofactors_in_dataset, index=y_test.index)

        y_pred_fd = y_pred_fd.add_prefix(name)

        y_test_pred = pd.concat([y_test, y_pred_fd], axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        hamming = hamming_loss(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        confusion = multilabel_confusion_matrix(y_test, y_pred)
        results[name] = {"accuracy": accuracy, "report": report, "confusion": confusion, "hamming_loss": hamming}

        print(results[name])

        return results, y_test_pred

    def save_results_to_file(self, results, filename):
        """
        Save the classification results to a file.

        Parameters:
        - results (dict): Dictionary containing the classification results.
        - filename (str): Name of the output file.

        Returns:
        - None
        """
        with open(filename, 'w') as file:
            for name, result in results.items():
                file.write(f"Classifier: {name}\n")
                file.write("-" * 20 + "\n") 
                if 'error' in result:
                    file.write(f"Error: {result['error']}\n\n")
                else:
                    file.write(f"Accuracy: {result['accuracy']:.3f}\n")
                    file.write(f"Hamming Loss: {result['hamming_loss']:.5f}\n")
                    file.write("Classification Report:\n")
                    file.write(f"{result['report']}\n")
                    file.write("Confusion Matrix:\n")

                    for matrix in result['confusion']:
                        np.savetxt(file, matrix, fmt="%d")
                        file.write("\n")
                    file.write("=" * 40 + "\n\n")
