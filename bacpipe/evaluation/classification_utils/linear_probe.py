import torch
import torch.nn as nn
import json
from torch.nn.functional import softmax

# define class linear_probe


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearProbe, self).__init__()
        self.lp = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lp(x)


def train_linear_probe(
    linear_probe, train_dataloader, task_config, device_str="cuda:0"
):

    # device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    linear_probe = linear_probe.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        linear_probe.parameters(), lr=task_config["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(task_config["num_epochs"]):
        linear_probe.train()
        print(f"Epoch {epoch+1}/{task_config['num_epochs']}")
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for embeddings, y in train_dataloader:
            embeddings, y = embeddings.to(device), y.to(device)

            # Forward pass through linear probe
            outputs = linear_probe(embeddings)

            # Compute loss
            loss = criterion(outputs, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            running_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += embeddings.size(0)
            correct_train += (predicted == y).sum().item()

        train_loss = running_loss / len(train_dataloader.dataset)
        train_accuracy = 100 * correct_train / total_train

        # print(f"Epoch {epoch + 1}/{task_config['num_epochs']}, Loss: {train_loss}, Accuracy: {train_accuracy}")

        print(
            f"Epoch [{epoch+1}/{task_config['num_epochs']}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%"
        )

    return linear_probe


def inference_with_linear_probe(linear_probe, test_dataloader, device_str="cuda:0"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    linear_probe = linear_probe.to(device)

    linear_probe.eval()
    y_pred = []
    y_true = []
    probabilities = []

    for embeddings, y in test_dataloader:
        embeddings, y = embeddings.to(device), y.to(device)

        outputs = linear_probe(embeddings)
        probs = softmax(outputs, dim=1).detach().cpu().numpy()

        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy().tolist())
        y_true.extend(y.cpu().numpy().tolist())
        probabilities.extend(probs)

    return y_pred, y_true, probabilities


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNN(nn.Module):
    def __init__(self, in_dim, out_dim, n_neighbors=15):
        super(KNN, self).__init__()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_trained = False  # Flag to track if KNN is trained

    def fit(self, x, y):
        """Train KNN classifier with numpy data"""
        x_np = x.cpu().detach().numpy()  # Convert tensor to NumPy
        y_np = y.cpu().detach().numpy()
        self.knn.fit(x_np, y_np)
        self.is_trained = True

    def forward(self, x):
        """Predict using KNN (only after it's trained)"""
        if not self.is_trained:
            raise ValueError("KNN model is not trained. Call `fit()` first.")

        x_np = x.cpu().detach().numpy()
        preds = self.knn.predict(x_np)  # Predict labels
        probs = self.knn.predict_proba(x_np)  # Predict probabilities

        preds_tensor = torch.tensor(preds, dtype=torch.long, device=x.device)
        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=x.device)

        return preds_tensor, probs_tensor


def train_knn_probe(knn_probe, train_dataloader, task_config, device_str="cpu"):
    device = torch.device(device_str)
    knn_probe.to(device)

    all_embeddings = []
    all_labels = []

    # Collect all embeddings and labels to train KNN
    for embeddings, y in train_dataloader:
        embeddings, y = embeddings.to(device), y.to(device)
        all_embeddings.append(embeddings)
        all_labels.append(y)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Train KNN
    knn_probe.fit(all_embeddings, all_labels)
    print("KNN Training Complete!")

    return knn_probe


def inference_with_knn_probe(knn_probe, test_dataloader, device_str="cpu"):
    device = torch.device(device_str)
    knn_probe.to(device)

    y_pred = []
    y_true = []
    probabilities = []

    for embeddings, y in test_dataloader:
        embeddings, y = embeddings.to(device), y.to(device)

        preds, probs = knn_probe(embeddings)  # Get predictions & probs

        y_pred.extend(preds.cpu().numpy().tolist())
        y_true.extend(y.cpu().numpy().tolist())
        probabilities.extend(probs.cpu().numpy().tolist())

    return y_pred, y_true, probabilities
