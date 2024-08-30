import json
import os
import time
import random
from tempfile import TemporaryDirectory
import numpy as np
import optuna
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import R2Score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix,hamming_loss, jaccard_score,coverage_error
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
# from preprocessing import get_reference_peak
model_name = 'flashprot_hpo.pth'


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def setUp():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = json.load(open("config.json"))["label_names"]
    r2_score = R2Score(num_outputs=len(labels)).to(device)
    return device, r2_score, labels


def load_data():
    dataset_name = "/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/flashprot.h5"
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
    # external_test.reset_index(inplace=True, drop=True)
    # X_external_test = external_test.iloc[:, :-1].drop(labels, axis=1)
    # y_external_test = external_test.iloc[:, :-1][labels]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train.values).unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.Tensor(y_train.values)
    X_val_tensor = torch.Tensor(X_val.values).unsqueeze(1)  # Add channel dimension
    y_val_tensor = torch.Tensor(y_val.values)
    X_test_tensor = torch.Tensor(X_test.values).unsqueeze(1)  # Add channel dimension
    y_test_tensor = torch.Tensor(y_test.values)
    # X_external_test_tensor = torch.Tensor(X_external_test.values).unsqueeze(1)  # Add channel dimension
    # y_external_test_tensor = torch.Tensor(y_external_test.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    n_samples, n_labels = train_dataset.tensors[1].shape
    class_counts = y_train_tensor.sum(dim=0)
    negatives_counts = n_samples - class_counts
    class_weights = negatives_counts / (class_counts + 1e-6)
    #class_weights = class_weights/class_weights.sum()




    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, worker_init_fn=np.random.seed(42))
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, worker_init_fn=np.random.seed(42))
    dataloaders = {"train": train_loader, "val": val_loader}
    spectra_shape = X_train_tensor.shape[2]
    dataset_sizes = {"train": len(X_train_tensor), "val": len(X_val_tensor), "test": len(X_test_tensor)}
    return (dataloaders, dataset_sizes, spectra_shape, X_test_tensor, y_test_tensor,
            None, None, None, labels, class_weights)  # , external_test.iloc[:, -1]) y_external_test_tensor


# Define the CNN model
class CNNModel(nn.Module):
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
        #x = self.bn4(self.relu(self.conv4(x)))
        x = x.view(-1, 128 * self.spectra_shape // 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class EarlyStopper:
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
    Fits an ensemble of CNN models.

    Args:
        n_members (int): The number of models in the ensemble.

    Returns:
        list: A list containing the trained models in the ensemble.
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


def predict_with_pi(ensemble, X):
    means, lower_bounds, upper_bounds = list(), list(), list()
    for row in X:
        yhat = [model(row) for model in ensemble]
        yhat = torch.stack(yhat)
        mean = yhat.mean(0)
        interval = 1.96 * yhat.std()
        lower, upper = yhat.mean() - interval, yhat.mean() + interval
        means.append(mean.item())
        lower_bounds.append(lower.item())
        upper_bounds.append(upper.item())
    return lower_bounds, means, upper_bounds


def bootstrap_prediction(model, input_data, device, n_bootstrap=1000, batch_size=100, confidence_level=0.70):
    input_data = input_data.to(device)  # Move input data to the device
    predictions = []
    lower_bounds = []
    upper_bounds = []
    for _ in range(n_bootstrap):
        for i in range(0, len(input_data), batch_size):  # Process the data in batches
            # Create a bootstrap sample of the data
            indices = torch.randint(0, len(input_data[i:i + batch_size]), (len(input_data[i:i + batch_size]),))
            bootstrap_sample = input_data[indices]

            # If the batch is smaller than the batch_size, pad it
            if bootstrap_sample.shape[0] < batch_size:
                padding = torch.zeros((batch_size - bootstrap_sample.shape[0], *bootstrap_sample.shape[1:])).to(device)
                bootstrap_sample = torch.cat([bootstrap_sample, padding])

            # Make predictions with the model
            with torch.no_grad():
                model.eval()
                prediction = model(bootstrap_sample)
                predictions.extend(prediction.tolist())

            # Calculate the mean and standard deviation of the predictions
            mean_prediction = prediction.mean(dim=0)
            std_prediction = prediction.std(dim=0)

            # Calculate the z-score for the desired confidence level
            z = scipy.stats.norm.ppf((1 + confidence_level) / 2)

            # Calculate the prediction intervals
            prediction_interval_lower = mean_prediction - z * std_prediction
            prediction_interval_upper = mean_prediction + z * std_prediction

            lower_bounds.extend(prediction_interval_lower.tolist())
            upper_bounds.extend(prediction_interval_upper.tolist())

    # Ensure all lists are of the same length
    min_length = min(len(predictions), len(lower_bounds), len(upper_bounds))
    predictions = predictions[:min_length]
    lower_bounds = lower_bounds[:min_length]
    upper_bounds = upper_bounds[:min_length]

    # Convert the predictions and prediction intervals to a DataFrame
    df = pd.DataFrame({
        'predictions': predictions,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds
    })

    return df


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, trial=None):
    
    since = time.time()
    # Create a temporary directory to save training checkpoints
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
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
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
                    # 'R2': batch_r2})
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
            # trial.report(epoch_r2, epoch)
            # if trial.should_prune():
            #    raise optuna.exceptions.TrialPruned()
      
        time_elapsed = time.time() - since
        plt.plot(training_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.legend()
        plt.savefig("/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/cnn/losses.png")
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val R2: {best_f1:4f}')
        with open("/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/cnn/best_f1.txt", "w") as f:
            f.write(str(best_f1))
        model.load_state_dict(torch.load(best_model_params_path))
    return model

def test_model(model, x_test_tensor, y_test_tensor, outlabels, metabolite=None, filename="predictions_test.tsv"):
    """
    Test the trained model on the test dataset and generate evaluation metrics.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        x_test_tensor (torch.Tensor): The input test data tensor.
        y_test_tensor (torch.Tensor): The target test data tensor.
        outlabels (list): The list of output labels.
        metabolite (str, optional): The name of the metabolite. Defaults to None.
        filename (str, optional): The name of the output file. Defaults to "predictions_test.tsv".

    Returns:
        float: The F1 score of the model on the test dataset.
    """
    sigmoid = nn.Sigmoid()
    batch_size = len(x_test_tensor)  # initially set batch size to the length of the test set
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
        with open('/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/cnn/results.txt', 'w') as f:
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
    # lr: 0.000512819169189802
    #    weight_decay: 5.720694936459442e-05
    #   step_size: 5
    #  gamma: 0.7264294814381911

    if params is None and trial is None:
        params = {"lr": 0.000512819169189802, "weight_decay": 5.720694936459442e-05, "step_size": 5, "gamma": 0.7264294814381911}
    elif params is None:
        lr = trial.suggest_float('lr', 1e-10, 1e-3)
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
    torch.save(model, "/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/cnn/bestflash_CNN.pth")
    f1_score = test_model(model, X_test_tensor, y_test_tensor, labels, filename = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/cnn/predictions_test.tsv')
    return model, f1_score


def hpo():
    from sklearn.model_selection import ParameterGrid
    param_grid = {
        'lr': [1e-3, 1e-4, 1e-5],
        'weight_decay': [1e-3, 1e-4, 1e-5],
        'step_size': [5, 7, 10],
        'gamma': [0.1, 0.5, 0.9]
    }

    # Create a grid of hyperparameters
    grid = ParameterGrid(param_grid)

    best_r2 = 0
    best_params = None

    # Iterate over each combination of hyperparameters
    for params in grid:
        model, r2 = run(params)

        # If the current model is better than the best model so far, update the best model and best hyperparameters
        if r2 > best_r2:
            best_r2 = r2
            best_params = params

    print(f'Best R2: {best_r2}')
    print(f'Best hyperparameters: {best_params}')


def hpo_optuna():
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
    f1_score = test_model(loaded_model, X_test_tensor, y_test_tensor, labels, filename = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/hpo/predictions_test.tsv')



if __name__ == "__main__":
    from torchmetrics.functional import f1_score, precision, recall, accuracy
    # data_path = join(get_config().get("PATHS", "OMNIACCPN_ROOT"), "data/concentrations/Full_Spectra")
    # os.chdir(data_path)
    # device, r2_score, labels = setUp()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (dataloaders, dataset_sizes, spectra_shape, X_test_tensor, y_test_tensor,
     X_external_test_tensor, y_external_test_tensor, metabolites_labels, labels, weights) = load_data()
    run()
    #model = torch.load("/home/jgoncalves/cofactor_prediction_tool/data/final_datasets/SeqVec/cnn/best_CNN.pth")
    # test_model(model, X_external_test_tensor, y_external_test_tensor, labels)
    # group external tensor by metabolite and test_model for each metabolite
    # for metabolite in metabolites_labels.unique():
    #     print(metabolite)
    #     mask = metabolites_labels == metabolite
    #     test_model(model, X_external_test_tensor[mask], y_external_test_tensor[mask], labels,
    #                metabolite=metabolite,
    #                filename=f"predictions_{metabolite}.tsv")

    # print(bootstrap_prediction(model, X_test_tensor, device, n_bootstrap=1))
    # ensemble = fit_ensemble(5)
    # ensemble = [torch.load(f"model_{i}.pth") for i in range(3)]
    # newX = X_test_tensor[0:10]
    # true_y = y_test_tensor[0:10].tolist()
    # lower, mean, upper = predict_with_pi(ensemble, newX)
    # for i in range(len(true_y)):
    #    print(f"True: {true_y[i]}, Predicted: {mean[i]}, Lower: {lower[i]}, Upper: {upper[i]}")

    # test_model(model, X_test_tensor, y_test_tensor, labels)
    # predict(device, labels)
    # hpo()
    #hpo_optuna()
    """
    Mean Absolute Error on Test Set: 0.1276257485151291, Test R^2: 0.9359, Test MSE: 0.18062952160835266,Test MAPE: 37.53916621208191"""
