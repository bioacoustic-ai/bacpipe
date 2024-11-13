import torch
import torch.nn as nn
import wandb

# define class linear_probe

class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)
    


def train_linear_probe(linear_probe, train_dataloader, configs):

    if configs["wandb_log"]:
        wandb.init(project=configs["wandb_project_name"]) 

    device = torch.device(configs["device"])
    linear_probe = linear_probe.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=configs["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(configs["num_epochs"]):
        linear_probe.train()
        for embeddings, y, labels, filename in train_dataloader:  
            # embeddings, labels = embeddings.to(device), labels.to(device)
            
            # Forward pass through linear probe
            outputs = linear_probe(embeddings)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # Track training loss and accuracy
            running_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_dataloader.dataset)
        train_accuracy = 100 * correct_train / total_train

        print(f"Epoch {epoch + 1}/{configs['num_epochs']}, Loss: {train_loss}, Accuracy: {train_accuracy}")
        
        if configs["wandb_log"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
            })


    return linear_probe





def inference_with_linear_probe(linear_probe, test_dataloader, configs):
    device = torch.device(configs["device"])
    linear_probe = linear_probe.to(device)

    linear_probe.eval()

    predictions = []

    for embeddings, labels in test_dataloader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        # Forward pass through linear probe
        outputs = linear_probe(embeddings)
        
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted.item())

    return predictions
