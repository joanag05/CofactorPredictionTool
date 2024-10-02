import os
import random
import time
from os.path import join
from tempfile import TemporaryDirectory
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, hamming_loss, multilabel_confusion_matrix
from sklearn.metrics import f1_score as f1_score_sklearn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import f1_score, precision, recall, accuracy
from tqdm import tqdm
import joblib
pd.set_option('display.max_columns', None)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name: str = None, model_path: str = None) -> torch.nn.Module:
    """
    Load a pre-trained model based on the provided model name.
    
    Args:
        model_name (str): The name of the model to load.
        model_path (str): The path to the model file to load.

    Returns:
        model (torch.nn.Module): The loaded PyTorch model.
    """
    print(os.getcwd())

    if model_name == 'cnn':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/bestesm2_hpo.pth')
        model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.eval()
    elif model_name == 'lb':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/LabelPowerset.joblib')
        model = joblib.load(model_path)

    elif model_name == 'xb':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/XGBoost.joblib')
        model = joblib.load(model_path)


    elif not model_name and model_path:
        pass

    else:
        raise ValueError("Invalid model name")
    return model


def load_data(dataset_path):
    """
    Load and preprocess the data for training a deep learning model.

    Returns:
        tuple: A tuple containing the following elements:
            - dataloaders (dict): A dictionary containing the train and validation data loaders.
            - dataset_sizes (dict): A dictionary containing the sizes of the train, validation, and test datasets.
            - dataset_shape (int): The shape of the dataset.
            - X_test_tensor (torch.Tensor): The test input data as a PyTorch tensor.
            - y_test_tensor (torch.Tensor): The test target data as a PyTorch tensor.
            - None: Placeholder for an unused variable.
            - None: Placeholder for an unused variable.
            - None: Placeholder for an unused variable.
            - labels (list): A list of column labels for the target data.
            - class_weights (torch.Tensor): The class weights for the target data.

    """
    x_train = pd.read_hdf(dataset_path, key='X_train', mode='r')
    y_train = pd.read_hdf(dataset_path, key='y_train', mode='r')
    x_val = pd.read_hdf(dataset_path, key='X_val', mode='r')
    y_val = pd.read_hdf(dataset_path, key='y_val', mode='r')
    x_test = pd.read_hdf(dataset_path, key='X_test', mode='r')
    y_test = pd.read_hdf(dataset_path, key='y_test', mode='r')
    train_data = pd.concat([x_train, y_train], axis=1)
    val_data = pd.concat([x_val, y_val], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    labels = y_train.columns.tolist()

    x_train = train_data.drop(labels, axis=1)
    y_train = train_data[labels]
    x_val = val_data.drop(labels, axis=1)
    y_val = val_data[labels]
    x_test = test_data.drop(labels, axis=1)
    y_test = test_data[labels]

    # Convert data to PyTorch tensors
    x_train_tensor = torch.Tensor(x_train.values).unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.Tensor(y_train.values)
    x_val_tensor = torch.Tensor(x_val.values).unsqueeze(1)  # Add channel dimension
    y_val_tensor = torch.Tensor(y_val.values)
    x_test_tensor = torch.Tensor(x_test.values).unsqueeze(1)  # Add channel dimension
    y_test_tensor = torch.Tensor(y_test.values)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    n_samples, n_labels = train_dataset.tensors[1].shape
    class_counts = y_train_tensor.sum(dim=0)
    negatives_counts = n_samples - class_counts
    class_weights = negatives_counts / (class_counts + 1e-6)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, worker_init_fn=np.random.seed(42))
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, worker_init_fn=np.random.seed(42))
    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_shape = x_train_tensor.shape[2]
    dataset_sizes = {"train": len(x_train_tensor), "val": len(x_val_tensor), "test": len(x_test_tensor)}
    return dataloaders, dataset_sizes, dataset_shape, x_test, x_test_tensor, y_test_tensor, labels, class_weights


class CNNModel(nn.Module):
    """
    Convolutional Neural Network model for cofactor prediction.
    Args:
        dataset_shape (int): The shape of the input dataset.
        n_outputs (int): The number of output labels. Default is 1.

    """

    def __init__(self, dataset_shape, n_outputs=1):
        super(CNNModel, self).__init__()
        self.dataset_shape = dataset_shape
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128 * self.dataset_shape // 4, 64)
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
        x = x.view(-1, 128 * self.dataset_shape // 4)
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


def train_model(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler, dataloaders: dict, dataset_sizes: dict,
                num_epochs: int = 25, **kwargs) -> torch.nn.Module:
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

    Returns:
        model (torch.nn.Module): The trained model.
    """
    device = kwargs.get('device', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
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
                progress_bar = tqdm(total=dataset_sizes[phase], desc=f'{phase} Epoch {epoch + 1}/{num_epochs}', colour="white", position=0, leave=True)
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
                progress_bar.set_postfix(
                    {f'{phase} loss': epoch_loss, f'{phase} F1': epoch_f1, f'{phase} Precision': epoch_precision, f'{phase} Recall': epoch_recall,
                     f'{phase} Accuracy': epoch_accuracy, })

                # deep copy the model
                if phase == 'val' and epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    torch.save(model.state_dict(), best_model_params_path)
                    if early_stopping.early_stop(1 - epoch_f1):
                        print("Early stopping")
                        break

        time_elapsed = time.time() - since

        if kwargs.get('plot', False):
            plt.plot(training_losses, label='Training Loss')
            plt.plot(validation_losses, label='Validation Loss')
            plt.legend()
            # plt.savefig("/home/emanuel/PythonProjects/CofactorPredictionTool/data/Final/ESM2/cnn/losses.png")
            plt.savefig(join(kwargs.get('data_path', "./"), "losses.png"))

        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val R2: {best_f1:4f}')
        with open(join(kwargs.get("data_path"), "best_f1.txt"), "w") as f:
            f.write(str(best_f1))
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def test_model(model, x_test_tensor, y_test_tensor, outlabels, filename="predictions_test.tsv", **kwargs):
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
    device = kwargs.get('device', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    sigmoid = nn.Sigmoid()
    batch_size = len(x_test_tensor) // 2
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
        test_f1_samples = f1_score_sklearn(all_labels, all_outputs > 0.5, average='samples')
        test_precision = precision(all_outputs, all_labels, task='multilabel', num_labels=13)
        test_recall = recall(all_outputs, all_labels, task='multilabel', num_labels=13)
        test_hamming_loss = hamming_loss(all_outputs, all_labels)
        test_accuracy = accuracy(all_outputs, all_labels, task='multilabel', num_labels=13)
        test_report = classification_report(all_labels, all_outputs, target_names=outlabels)
        confusion_mat = multilabel_confusion_matrix(all_labels, all_outputs)
        print(f'Test F1: {test_f1.item()}, Test Precision: {test_precision.item()}, Test Recall: {test_recall.item()}, Test Accuracy: {test_accuracy.item()}',
              f'Test Hamming Loss: {test_hamming_loss}', f'Test Report:', test_report, f'Confusion Matrix:', confusion_mat)

        with open(join(kwargs.get("data_path"), "results.txt"), "w") as f:
            f.write(f'Test F1: {test_f1.item()}, Test Precision: {test_precision.item()}, Test Recall: {test_recall.item()}, '
                    f'Test Accuracy: {test_accuracy.item()}, Test Hamming Loss: {test_hamming_loss}\n')
            f.write(f'Test Report: {test_report}\n')
            for matrix in confusion_mat:
                np.savetxt(f, matrix, fmt="%d")
                f.write("\n")
            f.write("=" * 40 + "\n\n")

        cols = [f"{outlabels[i]}_pred" for i in range(len(outlabels))] + [f"{outlabels[i]}_true" for i in range(len(outlabels))]
        df = pd.DataFrame(torch.cat((all_outputs, all_labels), dim=1).cpu().detach().numpy(), columns=cols)
        df = df.reindex(sorted(df.columns), axis=1)
        df.to_csv(join(kwargs.get('data_path','./'), filename), index=False, sep="\t")
        return test_f1, test_f1_samples


def run(params=None, trial=None, device=None, labels=None, dataset_shape=None, dataloaders=None, dataset_sizes=None, x_test_tensor=None, y_test_tensor=None,
        data_path: str = "./"):
    """
    Train the CNN model with specified or default hyperparameters.
    
    Args:
        params (dict, optional): Dictionary of hyperparameters. Default uses preset values.
        trial (optuna.trial.Trial, optional): Optuna trial object for hyperparameter optimization. Default is None.
        device (torch.device, optional): The device to run the model on. Default is CUDA if available.
        labels (list, optional): List of output labels. Default is None.
        dataset_shape (int): The shape of the input dataset.
        dataloaders (dict): Dictionary containing the data loaders for training and validation.
        dataset_sizes (dict): Dictionary containing sizes of the training and validation datasets.
        X_test_tensor (torch.Tensor): Tensor of input test data.
        y_test_tensor (torch.Tensor): Tensor of target test labels.
        data_path (str, optional): The path to save the model and results. Default is "./".

    
    Returns:
        Tuple:
            - model (torch.nn.Module): The trained CNN model.
            - f1_score (float): The F1 score of the trained model on the test dataset.

    Parameters
    ----------
    data_path
    y_test_tensor
    x_test_tensor
    dataset_sizes
    dataloaders
    labels
    device
    params
    params
    trial
    dataset_shape
    """
    if params is None and trial is None:
        params = {"lr": 1e-5, "weight_decay": 1e-5, "step_size": 5, "gamma": 0.5}
        plot = True

    elif params is None:
        lr = trial.suggest_float('lr', 1e-10, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-1, log=True)
        step_size = trial.suggest_int('step_size', 1, 10, step=1)
        gamma = trial.suggest_float('gamma', 0.1, 0.9, step=0.05)
        params = {"lr": lr, "weight_decay": weight_decay, "step_size": step_size, "gamma": gamma}
        plot = False

    model = CNNModel(dataset_shape=dataset_shape, n_outputs=len(labels))
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, params['step_size'], params['gamma'])
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=50, trial=trial, device=device,
                        data_path=data_path, plot=plot)
    torch.save(model, join(data_path, "best_cnn.pth"))
    f1_score_res, f1_score_sklearn_res = test_model(model, x_test_tensor, y_test_tensor, labels, data_path=data_path)
    return model, f1_score_res, f1_score_sklearn_res


def select_thresholds(x_test: pd.DataFrame, y_test_tensor: torch.Tensor, data_path: str, device=None, labels=None):
    """
    Determine the optimal threshold for the model's predictions to maximize the F1 score.
    
    This function loads a pre-trained model and a test dataset, then evaluates the model's
    performance across a range of prediction thresholds. It identifies and reports the threshold
    that results in the best F1 score.
    """
    model = torch.load(join(data_path, "best_cnn.pth"))
    thresholds = np.linspace(0.1, 0.9, 9)
    recalls = {}
    precisions = {}
    f1s = {}

    best_f1 = 0
    best_threshold = None

    for threshold in thresholds:

        y_pred = (torch.tensor(predict_proba(model, x_test, device, labels, batch_size=32).values) > threshold).float()

        recall_value = recall(y_pred, y_test_tensor, task='multilabel', num_labels=13)
        precision_value = precision(y_pred, y_test_tensor, task='multilabel', num_labels=13)
        f1_value = f1_score(y_pred, y_test_tensor, task='multilabel', num_labels=13)

        recalls[threshold] = recall_value.item()
        precisions[threshold] = precision_value.item()
        f1s[threshold] = f1_value.item()

        if f1_value > best_f1:
            best_f1 = f1_value
            best_threshold = threshold

    plt.plot(list(f1s.keys()), list(f1s.values()))
    plt.plot(list(recalls.keys()), list(recalls.values()))
    plt.plot(list(precisions.keys()), list(precisions.values()))
    plt.legend(['F1 Score', 'Recall', 'Precision'])
    plt.savefig(join(data_path, "f1_threshold.png"))
    print(f'Best F1: {best_f1}')
    print(f'Best threshold: {best_threshold}')


def hpo_optuna(device=None, labels=None, dataset_shape=None, dataloaders=None, dataset_sizes=None, x_test_tensor=None, y_test_tensor=None,
               model_name: str = None,  data_path="./"):
    """
    Perform hyperparameter optimization using Optuna.
    
    This function uses Optuna, a hyperparameter optimization framework, to maximize the F1 score
    of the model. It saves the best model and reports the best hyperparameters found during the search.
    """
    import optuna
    def objective(trial):
        model, f1_score_res, f1_score_sklearn_res = run(trial=trial, device=device, labels=labels, dataset_shape=dataset_shape, dataloaders=dataloaders,
                                                        dataset_sizes=dataset_sizes, x_test_tensor=x_test_tensor, y_test_tensor=y_test_tensor,
                                                        data_path=data_path)
        try:
            if f1_score_sklearn_res > trial.study.best_value:
                torch.save(model, join(data_path, model_name + ".pth"))
        except Exception as e:
            print(e)
            print("First trial")
        return f1_score_sklearn_res

    study = optuna.create_study(direction='maximize')
    study.enqueue_trial(params={"lr": 1e-5, "weight_decay": 1e-5, "step_size": 5, "gamma": 0.5})
    study.optimize(objective, n_trials=200)
    
    print('Best trial:')
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    model = load_model(model_path=join(data_path, model_name + ".pth"))
    f1_score_res, f1_score_sklearn_res = test_model(model, x_test_tensor, y_test_tensor, labels, data_path=data_path)


def predict(model: nn.Module, x: pd.DataFrame, original_labels: pd.DataFrame, device: torch.device = None, labels: list = None, threshold: float = 0.5):
    """
    Generate binary predictions for the input data using the trained model and include original labels.
    
    Args:
        model (torch.nn.Module): The trained model to use for prediction.
        x (pd.DataFrame): The input data tensor.
        original_labels (pd.DataFrame): The original labels DataFrame.
        device (torch.device, optional): The device to run the model on. Default is CUDA if available.
        labels (list, optional): List of output labels. Default is None.
        threshold (float, optional): Threshold to convert model outputs to binary labels. Default is 0.5.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the original labels and binary predictions.
    """
    sigmoid = nn.Sigmoid()
    batch_size = len(x)
    all_outputs = []
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataloader = DataLoader(TensorDataset(torch.tensor(x.values, dtype=torch.float)), batch_size=batch_size, shuffle=False)
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

        predictions_df = pd.DataFrame(all_outputs.numpy(), columns=[f"{label}_pred" for label in labels])
        predictions_df.index = x.index

        combined_df = pd.concat([original_labels, predictions_df], axis=1)

    return combined_df


def predict_proba(model: nn.Module, x: Union[torch.Tensor, pd.DataFrame], original_labels: pd.DataFrame, device: torch.device = None, labels: list = None, batch_size: int = None):
    """
    Generate probability predictions for the input data using the trained model and include original labels.
    
    Args:
        model (torch.nn.Module): The trained model to use for prediction.
        x (Union[torch.Tensor, pd.DataFrame]): The input data tensor or DataFrame.
        original_labels (pd.DataFrame): The original labels DataFrame.
        device (torch.device, optional): The device to run the model on. Default is CUDA if available.
        labels (list, optional): List of output labels. Default is None.
        batch_size (int, optional): The batch size for processing input data. Default is the length of X.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the original labels and predicted probabilities.
    """
    sigmoid = nn.Sigmoid()

    if batch_size is None:
        batch_size = len(x)

    all_outputs = []
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataloader = DataLoader(TensorDataset(torch.tensor(x.values, dtype=torch.float)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        model.eval()
        for batch in test_dataloader:
            inputs = batch[0].to(device)  # Access the tensor and move it to the device
            outputs = sigmoid(model(inputs))
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            all_outputs.append(outputs)

        all_outputs = torch.cat(all_outputs).cpu()

        if labels is None:
            labels = [f"label_{i}" for i in range(all_outputs.shape[1])]

        predictions_df = pd.DataFrame(all_outputs.numpy(), columns=[f"{label}_proba" for label in labels])
        predictions_df.index = x.index

        combined_df = pd.concat([original_labels, predictions_df], axis=1)

    return combined_df.round(3)