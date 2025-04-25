import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Linear classification layer.

        Parameters
        ----------
        in_dim : int
            number of input dimensions (dictated by embeddings)
        out_dim : int
            number of output dimensions (dictated by classes in ground truth)
        """
        super(LinearClassifier, self).__init__()
        self.clfier = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.clfier(x)


def train_linear_classifier(
    linear_classifier,
    train_dataloader,
    learning_rate,
    num_epochs,
    device="cuda:0",
    **kwargs,
):
    """
    Linear classification training pipeline. Hyperparameters are specified
    in settings.yaml file and passed to this function.

    Parameters
    ----------
    linear_classifier : object
        classification object
    train_dataloader : DataLoader object
        dataset loader to iterate over
    learning_rate : float
        learning rate
    num_epochs : int
        number of epochs for training
    device : str, optional
        'cpu' or 'cuda', by default "cuda:0"

    Returns
    -------
    object
        trained linear classificaion object
    """
    device = torch.device(device)
    linear_classifier = linear_classifier.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        linear_classifier.train()
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for embeddings, y in train_dataloader:
            embeddings, y = embeddings.to(device), y.to(device)

            # Forward pass through linear classifier
            outputs = linear_classifier(embeddings)

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

        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%"
        )

    return linear_classifier


def inference(classifier, test_dataloader, device="cuda:0", config="linear", **kwargs):
    """
    Perform inference using classifier.

    Parameters
    ----------
    classifier : object
        trained classification object
    test_dataloader : DataLoader object
        dataset iterator
    device : str, optional
        'cpu' or 'cuda', by default "cuda:0"
    config : str, optional
        type of classification, by default "linear"

    Returns
    -------
    list
        prediction values in ints corresponding to labels
    list
        ground truth values in ints
    np.array
        probabilities for each class and each embedding
    """
    device = torch.device(device)
    classifier = classifier.to(device)

    classifier.eval()
    y_pred = []
    y_true = []
    probabilities = []

    for embeddings, y in test_dataloader:
        embeddings, y = embeddings.to(device), y.to(device)

        outputs = classifier(embeddings)
        if config == "linear":
            # Use softmax to get probabilities
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            _, predicted = torch.max(outputs, 1)

        elif config == "knn":
            # KNN does not require softmax
            predicted, probs = outputs
            probs = probs.cpu().numpy().tolist()

        y_pred.extend(predicted.cpu().numpy().tolist())
        y_true.extend(y.cpu().numpy().tolist())
        probabilities.extend(probs)

    return y_pred, y_true, probabilities


class KNN(nn.Module):
    def __init__(self, n_neighbors=15, testing=False, **kwargs):
        """
        K-nearest neighbor classifier.

        Parameters
        ----------
        n_neighbors : int, optional
            hyperparameter specified in settings.yaml file, by default 15
        """
        super(KNN, self).__init__()
        if testing:
            n_neighbors = 1  # Set to 1 for testing purposes
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


def train_knn_classifier(knn_classifier, train_dataloader, device="cpu", **kwargs):
    """
    Pipeline for knn classifier training.

    Parameters
    ----------
    knn_classifier : object
        classifier object
    train_dataloader : DataLoader object
        iterator for dataset
    device : str, optional
        'cpu' or 'cuda', by default "cpu"

    Returns
    -------
    object
        classifier object
    """
    device = torch.device(device)
    knn_classifier.to(device)

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
    knn_classifier.fit(all_embeddings, all_labels)
    print("KNN Training Complete!")

    return knn_classifier
