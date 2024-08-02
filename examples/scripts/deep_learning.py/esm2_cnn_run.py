import torch
import optuna
import json
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

from cofactor_prediction_tool.deep_learning.cnn_esm import CNNModel, setUp, load_model, load_data, fit_ensemble, run, select_thresholds, hpo_optuna, EarlyStopper, train_model, test_model


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    # Initialize device and R2 score metric
    device, r2_score, labels = setUp()

    # Load data
    dataloaders, dataset_sizes, spectra_shape, X_test_tensor, y_test_tensor, _, _, _, labels, class_weights = load_data()

    # Run the model
    model, f1_score = run()

    # Save the best model
    torch.save(model, "best_esm2_model.pth")

    # Select the optimal thresholds
    select_thresholds()

    # Perform hyperparameter optimization using Optuna
    hpo_optuna()

if __name__ == "__main__":
    main()