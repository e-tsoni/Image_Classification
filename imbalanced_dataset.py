import optuna_dashboard
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as tt
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import optuna


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten the image
        out = self.linear(x)
        return out
        # return torch.sigmoid(out)


if __name__ == '__main__':

    train_tfms = tt.Compose([tt.ToTensor()])
    valid_tfms = tt.Compose([tt.ToTensor()])

    # Load train, validation and test dataset
    data_dir = os.getcwd()
    print("data_dir", data_dir)
    train_file = os.path.join(data_dir, "train")
    print("train_file", train_file)
    val_file = os.path.join(data_dir, "validation")
    print("val_file", val_file)
    test_file = os.path.join(data_dir, "test")
    print("test_file", test_file)

    train_ds = ImageFolder(train_file, train_tfms)
    val_ds = ImageFolder(val_file, valid_tfms)
    test_ds = ImageFolder(test_file, valid_tfms)

    # Batch Size
    batch_size = 1000
    epochs = 10

    # PyTorch data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, num_workers=3, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size, num_workers=3, pin_memory=True)

    input_size = 50 * 50 * 3  # Flattened size of images
    model = LogisticRegression(input_size)

    train_positive_samples = 57089
    train_negative_samples = 144435
    pos_weight = torch.tensor([train_negative_samples / train_positive_samples])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters())

    # epochs = 10
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for images, labels in train_dl:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        for images, labels in val_dl:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels.view(-1, 1).float())
                val_loss += loss.item()
                val_steps += 1
        avg_val_loss = val_loss / val_steps
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}')

        train_losses.append(epoch_train_loss / len(train_dl))
        val_losses.append(avg_val_loss)

    # Plotting
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_dl:
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels.view(-1, 1).float())
            predicted = torch.round(outputs)
            y_true.extend(labels.view(-1).tolist())
            y_pred.extend(predicted.view(-1).tolist())

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f'Test Loss: ', {loss})
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

    metrics = [precision, recall, f1_score, loss]
    metric_names = ['Precision', 'Recall', 'F1 Score', 'loss']

    plt.bar(metric_names, metrics)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Precision, Recall, F1 Score and Loss of the Logistic Regression Model')
    plt.ylim(0, 1)
    plt.show()
