import unittest
import torch
import sys
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")
from cofactor_prediction_tool.deep_learning import DeepCofactorNN
import os
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score

class TestDeepCofactor(unittest.TestCase):

    def test_dl(self):
        self.output_dim = 2
        X = torch.rand(100, 10)
        y = torch.randint(0, 2, (100, 2))
        X_test = torch.rand(10, 10)
        y_test = torch.randint(0, 2, (10, 2))
        dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
        X_val = torch.rand(10, 10)
        y_val = torch.randint(0, 2, (10, 2))
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        dataset = torch.utils.data.TensorDataset(X, y)
        model = DeepCofactorNN(10, 2)

        model.train_net(dataset,loss_kwargs={}, batch_size=5, num_epochs=2, epochs_wo_improve=5, val_dataset=val_dataset, time_limit=None, reporter=None, verbosity=2)

        accuracy = MultilabelAccuracy(num_labels=self.output_dim)
        precision = MultilabelPrecision(num_labels=self.output_dim)
        recall = MultilabelRecall(num_labels=self.output_dim)
        f1_score = MultilabelF1Score(num_labels=self.output_dim)
        evaluation = model.evaluate(dataset_test, [accuracy, precision, recall, f1_score])
        

        model.save_model("esm.pth")

      
     

        loaded_model = torch.load("esm.pth")
        loaded_model.eval()
        loaded_accuracy = loaded_model.evaluate(dataset_test, [accuracy, precision, recall, f1_score])
        self.assertEqual(evaluation, loaded_accuracy)
       


if __name__ == '__main__':
    unittest.main()