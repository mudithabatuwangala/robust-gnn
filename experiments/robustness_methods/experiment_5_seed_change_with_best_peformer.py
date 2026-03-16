import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

# -------------------------------
# Load dataset
# -------------------------------
dataset = TUDataset(root='data', name='Mutagenicity')
dataset = dataset.shuffle()
train_dataset = dataset[:3500]
test_dataset = dataset[3500:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------------------
# GCN Model
# -------------------------------
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dataset.num_node_features, hidden_channels))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# -------------------------------
# Training function
# -------------------------------
def train(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# -------------------------------
# Testing function with softmax
# -------------------------------
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        probs = F.softmax(out, dim=1)
        pred = probs.argmax(dim=1)
        correct += int((pred == data.y).sum())
    acc = correct / len(loader.dataset)
    return acc, probs, pred

# -------------------------------
# Run for multiple seeds
# -------------------------------
seeds = [1, 42, 123, 999, 2024]
hidden_channels = 128
num_layers = 5

for seed in seeds:
    torch.manual_seed(seed)
    model = GCN(hidden_channels=hidden_channels, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train for 10 epochs
    for epoch in range(10):
        train(model, train_loader, optimizer, criterion)
    
    # Evaluate
    acc, probs, pred = test(model, test_loader)
    
    print(f"\nRunning with seed: {seed}")
    print(f"Test Accuracy: {acc:.4f}")
    print("Softmax probabilities (first 5 molecules):")
    print(probs[:5])
    print("Predicted classes:", pred[:5])
    print("True labels:", test_dataset[:5].y)