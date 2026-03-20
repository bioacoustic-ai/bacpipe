import torch
import torch.nn as nn

from sklearn.neighbors import KNeighborsClassifier
from .dataset_probe import probe_dataset_loader

import logging

logger = logging.getLogger("bacpipe")


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
        self.probe = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.probe(x)


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
        trained linear classification object
    """
    device = torch.device(device)
    try:
        linear_classifier = linear_classifier.to(device)
    except RuntimeError:
        logger.error('Traceback', exc_info=True)
        logger.info(
            "This problem is likely cause by tensorflow being a pain in the ****. "
            "The best fix for this is to simply restart bacpipe with the same settings, "
            "that way the GPU should be available for pytorch. Alternatively select "
            "`cpu` for device in the settings.yaml file."
        )
        import sys
        sys.exit(0)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        linear_classifier.train()
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
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

        # logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%"
        )

    return linear_classifier




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
            error = ("\nKNN model is not trained. Call `fit()` first.")
            logger.exception(error)
            raise ValueError(error)

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
    logger.info("KNN Training Complete!")

    return knn_classifier



def train_classifier(embeds, df, label2index, config="linear", **kwargs):
    """
    Classification pipeline. First the classification dataframe is loaded,
    then a dict is created to link labels to ints, then the dataset loaders
    are created to iterate over. Next depending of the specified config
    a linear or KNN classification is performed. Finally the classifiers are
    used for inference and based on that performance metrics are created.

    Parameters
    ----------
    paths : SimpleNamespace dict
        dictionary object containing paths for loading and saving
    dataset_csv_path : string
        name of classification dataframe as secified in the settings.yaml file
    embeds : np.array
        the embeddings
    config : str, optional
        type of classification, by default 'linear'

    Returns
    -------
    dict
        performance dictionary
    """

    # generate the loaders
    train_gen = probe_dataset_loader("train", df, embeds, label2index, **kwargs)

    embed_size = embeds[0].shape[-1]

    if config == "linear":
        probe = LinearClassifier(in_dim=embed_size, out_dim=len(df.label.unique()))
        probe = train_linear_classifier(probe, train_gen, **kwargs)

    elif config == "knn":
        if len(df[df.predefined_set =='test']) < kwargs['n_neighbors']:
            kwargs['n_neighbors'] = len(df[df.predefined_set =='test']) - 1
        probe = KNN(**kwargs)
        probe = train_knn_classifier(probe, train_gen, **kwargs)

    return probe
