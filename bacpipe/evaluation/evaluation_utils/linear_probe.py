import torch
import torch.nn as nn
import wandb

class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)



def train_linear_probe(linear_probe, train_dataloader, configs):   #getting data from dataloader is set for ID task only! for now...

    if configs["wandb_log"]:
        wandb.init(project=configs["wandb_project_name"], settings=wandb.Settings(init_timeout=120))

    device = torch.device(configs["device"])


    linear_probe = linear_probe.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=configs["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(configs["num_epochs"]):
        linear_probe.train()
        print(f"Epoch {epoch+1}/{configs['num_epochs']}")

        running_loss = 0
        correct_train = 0
        total_train = 0

        for embeddings, y, labels, filename in train_dataloader:
            embeddings = embeddings.to(device)

            # print(labels)
            # print(y)
            # print(embeddings.shape)
            y = y.to(device)

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

        if configs["wandb_log"]:
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
            })

        print(f"Epoch [{epoch+1}/{configs['num_epochs']}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    torch.save(linear_probe.state_dict(), "linear_probe.pth")
    return linear_probe





def inference_with_linear_probe(linear_probe, test_dataloader, configs):
    device = torch.device(configs["device"])
    linear_probe = linear_probe.to(device)

    linear_probe.eval()

    predictions = []

    for embeddings, y, labels, filenames in test_dataloader:
        embeddings, y = embeddings.to(device), y.to(device)

        # Forward pass through linear probe
        outputs = linear_probe(embeddings)

        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted.item())

    return predictions, y
