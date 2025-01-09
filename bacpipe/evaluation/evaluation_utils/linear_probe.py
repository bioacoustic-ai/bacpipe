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

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
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

        for embeddings, y, labels, filename in train_dataloader:
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

    for embeddings, y, _, _ in test_dataloader:
        embeddings, y = embeddings.to(device), y.to(device)

        outputs = linear_probe(embeddings)
        probs = softmax(outputs, dim=1).detach().cpu().numpy()

        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy().tolist())
        y_true.extend(y.cpu().numpy().tolist())
        probabilities.extend(probs)

    return y_pred, y_true, probabilities
