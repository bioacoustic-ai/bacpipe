import torch
import torch.nn as nn
import wandb
import json

# define class linear_probe

class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)
    


def train_linear_probe(linear_probe, train_dataloader, configs, device_str, wandb_configs_path=''):

    if wandb_configs_path:
        wandb_configs = json.load(open(wandb_configs_path, 'r'))
        wandb.init(project=wandb_configs["wandb_project_name"]) 

    device = torch.device(device_str)
    linear_probe = linear_probe.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=configs["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(configs["num_epochs"]):
        linear_probe.train()
        print(f"Epoch {epoch+1}/{configs['num_epochs']}")
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

        print(f"Epoch {epoch + 1}/{configs['num_epochs']}, Loss: {train_loss}, Accuracy: {train_accuracy}")
        
        if wandb_configs_path:
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
            })
        print(f"Epoch [{epoch+1}/{configs['num_epochs']}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        # if wandb_configs['save_trained_lp']:
        #     torch.save(linear_probe.state_dict(), wandb_configs["save_checkpoint_path"]+"_linear_probe.pt")
        
    return linear_probe





def inference_with_linear_probe(linear_probe, test_dataloader, device_str):
    device = torch.device(device_str)
    linear_probe = linear_probe.to(device)

    linear_probe.eval()
    predictions = []
    gt_indexes = []



    for embeddings, y, _ , _ in test_dataloader:
        embeddings, y = embeddings.to(device), y.to(device)
        
        outputs = linear_probe(embeddings)
        
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted.item())
        gt_indexes.append(y.item())

    return predictions, gt_indexes
