import shutil
from os.path import join

import torch

from cofactor_prediction_tool.deep_learning.cnn_model import load_data, run, select_thresholds, hpo_optuna
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = r"/home/emanuel/PythonProjects/CofactorPredictionTool/data/Final/SeqVec/seqvec.h5"
    (dataloaders, dataset_sizes, dataset_shape, X_test, X_test_tensor, y_test_tensor, labels, weights) = load_data(dataset_path)
    data_path = "/home/emanuel/PythonProjects/CofactorPredictionTool/data/Final/SeqVec/cnn"
    # Run the model
    model, f1_score, f1_score_sklearn_res = run(device=device, labels=labels, dataset_shape=dataset_shape, dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                                                x_test_tensor=X_test_tensor, y_test_tensor=y_test_tensor, data_path=data_path)
    print(f1_score_sklearn_res)
    # Save the best model
    torch.save(model, "best_seqvec_model.pth")

    # Select the optimal thresholds
    select_thresholds(X_test, X_test_tensor, data_path=data_path, device=device, labels=labels)

    # Perform hyperparameter optimization using Optuna
    hpo_optuna(device=device, labels=labels, dataset_shape=dataset_shape, dataloaders=dataloaders, dataset_sizes=dataset_sizes, x_test_tensor=X_test_tensor,
               y_test_tensor=y_test_tensor, model_name="bestseqvec_hpo", data_path=data_path)

    shutil.copy(join(data_path, "bestseqvec_hpo.pth"), "../../../src/cofactor_prediction_tool/resources/")


if __name__ == "__main__":
    main()
