import json
import os
import time
import random
from tempfile import TemporaryDirectory
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import R2Score
from tqdm import tqdm
import numpy as np
pd.set_option('display.max_columns', None)
from sklearn.metrics import classification_report, multilabel_confusion_matrix,hamming_loss, f1_score, multilabel_confusion_matrix


import optuna
import matplotlib.pyplot as plt

model_name = 'esm2_hpo.pth'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setUp():
    """
    Set up the directory paths for training and validation data, 
    and initialize the number of workers for data loading.

    Returns:
        Tuple:
            - train_dir (str): Path to the training data directory.
            - valid_dir (str): Path to the validation data directory.
            - num_workers (int): Number of workers to use for data loading.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = json.load(open("config.json"))["label_names"]
    r2_score = R2Score(num_outputs=len(labels)).to(device)
    return device, r2_score, labels

def load_model(model_name):
    """
    Load a pre-trained model based on the provided model name.
    
    Args:
        model_name (str): The name of the model to load.

    Returns:
        model (torch.nn.Module): The loaded PyTorch model.
    """
    print(os.getcwd())
    if model_name == 'esm2':
        modelfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../resources/bestesm2_hpo.pth')
    else:
        raise ValueError("Invalid model name")
    model = torch.load(modelfile)
    model.eval()
    return model

def load_data():
    """
    Load and preprocess the data for training a deep learning model.

    Returns:
        tuple: A tuple containing the following elements:
            - dataloaders (dict): A dictionary containing the train and validation data loaders.
            - dataset_sizes (dict): A dictionary containing the sizes of the train, validation, and test datasets.
            - spectra_shape (int): The shape of the spectra data.
            - X_test_tensor (torch.Tensor): The test input data as a PyTorch tensor.
            - y_test_tensor (torch.Tensor): The test target data as a PyTorch tensor.
            - None: Placeholder for an unused variable.
            - None: Placeholder for an unused variable.
            - None: Placeholder for an unused variable.
            - labels (list): A list of column labels for the target data.
            - class_weights (torch.Tensor): The class weights for the target data.

    """
    dataset_name = "/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2/esm2.h5"
    X_train = pd.read_hdf(dataset_name, key='X_train', mode='r')
    y_train = pd.read_hdf(dataset_name, key='y_train', mode='r')
    X_val = pd.read_hdf(dataset_name, key='X_val', mode='r')
    y_val = pd.read_hdf(dataset_name, key='y_val', mode='r')
    X_test = pd.read_hdf(dataset_name, key='X_test', mode='r')
    y_test = pd.read_hdf(dataset_name, key='y_test', mode='r')
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    labels = y_train.columns.tolist()

    X_train = train_data.drop(labels, axis=1)
    y_train = train_data[labels]
    X_val = val_data.drop(labels, axis=1)
    y_val = val_data[labels]
    X_test = test_data.drop(labels, axis=1)
    y_test = test_data[labels]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train.values).unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.Tensor(y_train.values)
    X_val_tensor = torch.Tensor(X_val.values).unsqueeze(1)  # Add channel dimension
    y_val_tensor = torch.Tensor(y_val.values)
    X_test_tensor = torch.Tensor(X_test.values).unsqueeze(1)  # Add channel dimension
    y_test_tensor = torch.Tensor(y_test.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    n_samples, n_labels = train_dataset.tensors[1].shape
    class_counts = y_train_tensor.sum(dim=0)
    negatives_counts = n_samples - class_counts
    class_weights = negatives_counts / (class_counts + 1e-6)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, worker_init_fn=np.random.seed(42))
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, worker_init_fn=np.random.seed(42))
    dataloaders = {"train": train_loader, "val": val_loader}
    spectra_shape = X_train_tensor.shape[2]
    dataset_sizes = {"train": len(X_train_tensor), "val": len(X_val_tensor), "test": len(X_test_tensor)}
    return (dataloaders, dataset_sizes, spectra_shape, X_test_tensor, y_test_tensor,
            None, None, None, labels, class_weights)


class CNNModel(nn.Module):
    """
    Convolutional Neural Network model for cofactor prediction.
    
    Attributes:
        features (nn.Module): Feature extraction layers.
        classifier (nn.Module): Classification layers.
    """

    def __init__(self, spectra_shape, n_outputs=1):
        super(CNNModel, self).__init__()
        self.spectra_shape = spectra_shape
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(128 * self.spectra_shape // 4, 64)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, n_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(self.bn1(self.relu(self.conv1(x))))
        x = self.pool(self.bn2(self.relu(self.conv2(x))))
        x = self.bn3(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * self.spectra_shape // 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class EarlyStopper:
    """
    Early stopping utility to stop training when the validation loss stops improving.

    Args:
        patience (int): How many epochs to wait after the last time the validation loss improved.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.

    Attributes:
        patience (int): How many epochs to wait before stopping.
        min_delta (float): Minimum change to qualify as an improvement.
        counter (int): Counts epochs where no improvement is seen.
        best_score (float): The best score observed so far.
        early_stop (bool): Flag to indicate whether early stopping should be triggered.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def fit_ensemble(n_members):
    """
    Fit an ensemble of models by training multiple instances.

    Args:
        n_members (int): Number of models to include in the ensemble.

    Returns:
        models (list): List of trained models.
    """
    ensemble = list()
    for i in range(n_members):
        params = {"lr": 1e-4, "weight_decay": 1e-3, "step_size": 9, "gamma": 0.27}
        model = CNNModel(spectra_shape=spectra_shape, n_outputs=len(labels))
        if torch.cuda.is_available():
            model.cuda()
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        criterion = nn.BCEWithLogitsLoss()
        # optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, params['step_size'], params['gamma'])
        model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=20)
        r2_test, mae = test_model(model, X_test_tensor, y_test_tensor, labels)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        model.cpu()
        torch.cuda.empty_cache()
        ensemble.append(model)
        torch.save(model, f"model_{i}.pth")
    return ensemble

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, trial=None):
    """
    Train a CNN model with a specified number of epochs and early stopping.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        dataloaders (dict): Dictionary containing the data loaders for training and validation.
        dataset_sizes (dict): Dictionary containing sizes of the training and validation datasets.
        num_epochs (int, optional): The number of epochs to train. Default is 25.
        patience (int, optional): The number of epochs to wait for improvement before stopping early. Default is 10.

    Returns:
        model (torch.nn.Module): The trained model.
        best_model_wts (dict): The weights of the model at the epoch with the best validation performance.
        history (dict): Dictionary containing the training and validation loss history.
    """

    since = time.time()
    early_stopping = EarlyStopper(patience=5)
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_f1 = 0.0
        training_losses = []
        validation_losses = []
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                total_batches = 0
                epoch_outputs, epoch_labels = [], []
                progress_bar = tqdm(total=dataset_sizes[phase], desc=f'{phase} Epoch {epoch + 1}/{num_epochs}',
                                    colour="white",
                                    position=0, leave=True)
                # Iterate over data.
                for i, (inputs, label_vals) in enumerate(dataloaders[phase]):
                    if inputs.size(0) < 2:
                        continue
                    inputs = inputs.to(device)
                    label_vals = label_vals.to(device).squeeze()
                    epoch_labels.append(label_vals)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs).squeeze()
                        loss = criterion(outputs, label_vals)
                        epoch_outputs.append(outputs)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    total_batches += 1
                    # statistics
                    batch_loss = loss.item()
                    running_loss += batch_loss

                    progress_bar.set_postfix({'loss': batch_loss})
                    progress_bar.update(inputs.size(0))

                epoch_loss = running_loss / total_batches

                if phase == 'train':
                    scheduler.step()

                    training_losses.append(epoch_loss)
                else:
                    validation_losses.append(epoch_loss)

                epoch_loss = running_loss / total_batches
                epoch_f1 = f1_score(torch.cat(epoch_outputs), torch.cat(epoch_labels), task='multilabel', num_labels=13).item()
                epoch_precision = precision(torch.cat(epoch_outputs), torch.cat(epoch_labels), task='multilabel', num_labels=13).item()
                epoch_recall = recall(torch.cat(epoch_outputs), torch.cat(epoch_labels), task='multilabel', num_labels=13).item()
                epoch_accuracy = accuracy(torch.cat(epoch_outputs), torch.cat(epoch_labels), task='multilabel', num_labels=13).item()
                progress_bar.set_postfix({f'{phase} loss': epoch_loss,
                                          f'{phase} F1': epoch_f1,
                                            f'{phase} Precision': epoch_precision,
                                            f'{phase} Recall': epoch_recall,
                                            f'{phase} Accuracy': epoch_accuracy,})

                # deep copy the model
                if phase == 'val' and epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    torch.save(model.state_dict(), best_model_params_path)
                    if early_stopping.early_stop(1 - epoch_f1):
                        print("Early stopping")
                        break

        time_elapsed = time.time() - since
        plt.plot(training_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.legend()
        plt.savefig("/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2/cnn/losses.png")
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val R2: {best_f1:4f}')
        with open("/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2/cnn/best_f1.txt", "w") as f:
            f.write(str(best_f1))
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def test_model(model, x_test_tensor, y_test_tensor, outlabels, metabolite=None, filename="predictions_test.tsv"):
    """
    Evaluate the trained model on a test dataset and generate evaluation metrics.
    
    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        x_test_tensor (torch.Tensor): Tensor of input test data.
        y_test_tensor (torch.Tensor): Tensor of target test labels.
        outlabels (list): List of output labels.
        metabolite (str, optional): Name of the metabolite (if applicable). Default is None.
        filename (str, optional): Name of the output file for predictions. Default is "predictions_test.tsv".
    
    Returns:
        float: The F1 score of the model on the test dataset.
    """
    sigmoid = nn.Sigmoid()
    batch_size = len(x_test_tensor)  
    all_outputs = []
    all_labels = []

    counts_per_label = {}
    for i in range(len(outlabels)):
        counts_per_label[outlabels[i]] = y_test_tensor[:, i].sum().item()
    print(counts_per_label)

    test_dataloader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        model.eval()
        for inputs, label_val in test_dataloader:
                outputs = sigmoid(model(inputs.to(device)))
                outputs = (outputs > 0.5).float()
                label_val = label_val.to(device)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if label_val.dim() == 0:
                    label_val = label_val.unsqueeze(0)
                all_outputs.append(outputs)
                all_labels.append(label_val)
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        all_outputs = all_outputs.cpu()
        all_labels = all_labels.cpu()

        test_f1 = f1_score(all_outputs, all_labels, task='multilabel', num_labels=13)
        test_precision = precision(all_outputs, all_labels, task='multilabel', num_labels=13)
        test_recall = recall(all_outputs, all_labels, task='multilabel', num_labels=13)
        test_hamming_loss = hamming_loss(all_outputs, all_labels)
        test_accuracy = accuracy(all_outputs, all_labels, task='multilabel', num_labels=13)
        test_report = classification_report(all_labels, all_outputs,target_names=outlabels)
        confusion_mat = multilabel_confusion_matrix(all_labels, all_outputs)
        print(f'Test F1: {test_f1.item()}, Test Precision: {test_precision.item()}, Test Recall: {test_recall.item()}, Test Accuracy: {test_accuracy.item()}',
            f'Test Hamming Loss: {test_hamming_loss}', f'Test Report:', test_report, f'Confusion Matrix:', confusion_mat)
        with open('/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2/cnn/results.txt', 'w') as f:
            f.write(f'Test F1: {test_f1.item()}, Test Precision: {test_precision.item()}, Test Recall: {test_recall.item()}, Test Accuracy: {test_accuracy.item()}, Test Hamming Loss: {test_hamming_loss}\n')
            f.write(f'Test Report: {test_report}\n')
            for matrix in confusion_mat:
                np.savetxt(f, matrix, fmt="%d")
                f.write("\n")
            f.write("=" * 40 + "\n\n")


        cols = [f"{outlabels[i]}_pred" for i in range(len(outlabels))] + [f"{outlabels[i]}_true" for i in range(len(outlabels))]
        df = pd.DataFrame(torch.cat((all_outputs, all_labels), dim=1).cpu().detach().numpy(), columns=cols)
        df = df.reindex(sorted(df.columns), axis=1)
        if metabolite:
            filename = f"predictions_{metabolite}.tsv"
        df.to_csv(filename, index=False, sep="\t")
        return test_f1
    
    
   

def run(params=None, trial=None):
    """
    Train the CNN model with specified or default hyperparameters.
    
    Args:
        params (dict, optional): Dictionary of hyperparameters. Default uses preset values.
        trial (optuna.trial.Trial, optional): Optuna trial object for hyperparameter optimization. Default is None.
    
    Returns:
        Tuple:
            - model (torch.nn.Module): The trained CNN model.
            - f1_score (float): The F1 score of the trained model on the test dataset.
    """
    if params is None and trial is None:
        params = {"lr": 0.000512819169189802, "weight_decay": 5.720694936459442e-05, "step_size": 5, "gamma": 0.7264294814381911}
    elif params is None:
        lr = trial.suggest_float('lr', 1e-10, 1e-1)
        weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-2)
        step_size = trial.suggest_int('step_size', 5, 10)
        gamma = trial.suggest_float('gamma', 0.1, 0.9)
        params = {"lr": lr, "weight_decay": weight_decay, "step_size": step_size, "gamma": gamma}
    model = CNNModel(spectra_shape=spectra_shape, n_outputs=len(labels))
    if torch.cuda.is_available():
        model.cuda()

    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, params['step_size'], params['gamma'])
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=50, trial=trial)
    torch.save(model, "/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2/cnn/bestesm_CNN.pth")
    f1_score = test_model(model, X_test_tensor, y_test_tensor, labels, filename = '/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2/cnn/predictions_test.tsv')
    return model, f1_score

def select_thresholds():
    """
    Determine the optimal threshold for the model's predictions to maximize the F1 score.
    
    This function loads a pre-trained model and a test dataset, then evaluates the model's
    performance across a range of prediction thresholds. It identifies and reports the threshold
    that results in the best F1 score.
    """
    dataset_name = "/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2/esm2.h5"    
    X_test = pd.read_hdf(dataset_name, key='X_test', mode='r')
    y_test = pd.read_hdf(dataset_name, key='y_test', mode='r')


    model = torch.load("/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2/hpo/bestesm2_hpo.pth")
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1 = 0
    best_recall = 0
    best_precision = 0
    best_threshold = 0

    recalls = {}
    precisions = {}
    f1s = {}

    best_f1 = 0
    best_threshold = None

    for threshold in thresholds:

        y_pred = (torch.tensor(predict_proba(model, X_test, device, labels, batch_size=32).values) > threshold).float()

        recall_value = recall(y_pred, y_test_tensor, task='multilabel', num_labels=13)
        precision_value = precision(y_pred, y_test_tensor, task='multilabel', num_labels=13)
        f1_value = f1_score(y_pred, y_test_tensor, task='multilabel', num_labels=13)
        

        recalls[threshold] = recall_value
        precisions[threshold] = precision_value
        f1s[threshold] = f1_value
        

        if f1_value > best_f1:
            best_f1 = f1_value
            best_threshold = threshold



        
    
    plt.plot(f1s.keys(), f1s.values())
    plt.plot(recalls.keys(), recalls.values())
    plt.plot(precisions.keys(), precisions.values())
    plt.legend(['F1 Score', 'Recall', 'Precision'])
    plt.savefig("/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2/cnn/f1_threshold.png")
    print(f'Best F1: {best_f1}')
    print(f'Best threshold: {best_threshold}')

def hpo_optuna():
    """
    Perform hyperparameter optimization using Optuna.
    
    This function uses Optuna, a hyperparameter optimization framework, to maximize the F1 score
    of the model. It saves the best model and reports the best hyperparameters found during the search.
    """
    def objective(trial):
        model, f1_score = run(trial=trial)
        try:
            if f1_score > trial.study.best_value:
                torch.save(model, 'best' + model_name)
        except:
            pass
        return f1_score
        

    study = optuna.create_study(direction='maximize')
    study.enqueue_trial(params = {"lr": 0.000512819169189802, "weight_decay": 5.720694936459442e-05, "step_size": 5, "gamma": 0.7264294814381911})
    study.optimize(objective, n_trials=100)

    print('Best trial:')
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    loaded_model = torch.load('best' +  model_name)
    f1_score = test_model(loaded_model, X_test_tensor, y_test_tensor, labels, filename = '/home/jgoncalves/cofactor_prediction_tool/data/Final/ESM2/hpo/predictions_test.tsv')



def predict(model, X, device= None, labels=None, threshold=0.5):
    """
    Generate binary predictions for the input data using the trained model.
    
    Args:
        model (torch.nn.Module): The trained model to use for prediction.
        X (torch.Tensor): The input data tensor.
        device (torch.device, optional): The device to run the model on. Default is CUDA if available.
        labels (list, optional): List of output labels. Default is None.
        threshold (float, optional): Threshold to convert model outputs to binary labels. Default is 0.5.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the binary predictions.
    """
    sigmoid = nn.Sigmoid()
    batch_size = len(X)  
    all_outputs = []
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_dataloader = DataLoader(TensorDataset(torch.tensor(X.values, dtype=torch.float).unsqueeze(1)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        model.eval()
        for batch in test_dataloader:
            inputs = batch[0].to(device)  # Access the tensor and move it to the device
            outputs = sigmoid(model(inputs))
            outputs = (outputs > threshold).float()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
               
            all_outputs.append(outputs)
    
        all_outputs = torch.cat(all_outputs)
        all_outputs = all_outputs.cpu()

        if labels is None:
            labels = [f"label_{i}" for i in range(all_outputs.shape[1])]

        df = pd.DataFrame(all_outputs, columns=labels)
        df.index = X.index

        return df
    
def predict_proba(model, X, device= None, labels=None, batch_size=None):
    """
    Generate probability predictions for the input data using the trained model.
    
    Args:
        model (torch.nn.Module): The trained model to use for prediction.
        X (torch.Tensor): The input data tensor.
        device (torch.device, optional): The device to run the model on. Default is CUDA if available.
        labels (list, optional): List of output labels. Default is None.
        batch_size (int, optional): The batch size for processing input data. Default is the length of X.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the predicted probabilities.
    """
    sigmoid = nn.Sigmoid()

    if batch_size is None:
        batch_size = len(X)  

    all_outputs = []
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_dataloader = DataLoader(TensorDataset(torch.tensor(X.values, dtype=torch.float).unsqueeze(1)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        model.eval()
        for batch in test_dataloader:
            inputs = batch[0].to(device)  # Access the tensor and move it to the device
            outputs = sigmoid(model(inputs))
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
               
            all_outputs.append(outputs)
            
        all_outputs = torch.cat(all_outputs)
        all_outputs = all_outputs.cpu()

        if labels is None:
            labels = [f"label_{i}" for i in range(all_outputs.shape[1])]

        df = pd.DataFrame(all_outputs, columns=labels)
        df.index = X.index

        return df.round(3)     

if __name__ == "__main__":
    from torchmetrics.functional import f1_score, precision, recall, accuracy
    # data_path = join(get_config().get("PATHS", "OMNIACCPN_ROOT"), "data/concentrations/Full_Spectra")
    # os.chdir(data_path)
    # device, r2_score, labels = setUp()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (dataloaders, dataset_sizes, spectra_shape, X_test_tensor, y_test_tensor,
     X_external_test_tensor, y_external_test_tensor, metabolites_labels, labels, weights) = load_data()
    print(labels)
    #select_thresholds()
    run()
    hpo_optuna()

