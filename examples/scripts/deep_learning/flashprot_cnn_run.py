import shutil
from os.path import join
import os
import torch
import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")
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
    print(os.getcwd())
    os.chdir(r"/home/jgoncalves/cofactor_prediction_tool")
    dataset_path = r"data/ml_dl_data/Flash/flashprot.h5"
    (dataloaders, dataset_sizes, dataset_shape, X_test, X_test_tensor, y_test_tensor, labels, weights) = load_data(dataset_path)
    data_path = "data/Final/ml_dl_data/Flash/bu"
    # Run the model
    model, f1_score, f1_score_sklearn_res = run(device=device, labels=labels, dataset_shape=dataset_shape, dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                                                x_test_tensor=X_test_tensor, y_test_tensor=y_test_tensor, data_path=data_path)
    print(f1_score_sklearn_res)
    # Save the best model
    torch.save(model, "best_flashprot_model.pth")

    # Select the optimal thresholds
    select_thresholds(X_test, y_test_tensor, data_path=data_path, device=device, labels=labels)


    shutil.copy(join(data_path, "bestflashprot_hpo.pth"), "src/cofactor_prediction_tool/resources2/")


if __name__ == "__main__":
    main()