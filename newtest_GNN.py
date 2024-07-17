import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from fairness import cluster_scanner

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load citation network dataset
dataset = Planetoid(root='data', name='Cora', split='full')
data = dataset[0].to(device)

# Define GNN model
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model, optimizer, and loss function
model = Net(dataset.num_features, 64, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Create train-test split using DataLoader
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)

# Use the test split for testing
test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluate the model
def test(loader):
    model.eval()
    for batch in loader:
        batch = batch.to(device)  # Move the entire batch to the same device
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        acc = pred[batch.test_mask].eq(batch.y[batch.test_mask]).sum().item() / batch.test_mask.sum().item()
        return acc

# Training with clustering
def train_with_clustering(num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)  # Move the entire batch to the same device
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

        # Perform clustering at the end of each epoch
        X =model(data.x, data.edge_index).detach().cpu().numpy()
        baseline = np.random.rand(*X.shape) # Replace this with an appropriate baseline

        # Convert to PyTorch tensors and move them to the same device
        X = torch.tensor(X, device=device)
        baseline = torch.tensor(baseline, device=device)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            points_orig_idx, _, _, _ = cluster_scanner(X, baseline, device=device)

        # Use points_orig_idx for further analysis or adjustment of the training data

        # Evaluate the model
        test_acc = test(test_loader)
        print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}, Test Accuracy: {test_acc:.4f}')

train_with_clustering(num_epochs=30)

    # main program or function calls go here

# Train the model with clustering

