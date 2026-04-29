import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
import copy
import numpy as np

# 1. LOAD DATASET
dataset = TUDataset(root='/tmp/Mutagenicity', name='Mutagenicity').shuffle()

# 2. DEFINE MODEL
class RobustGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(RobustGNN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = global_max_pool(x, batch)
        return self.lin(x)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# --- RUN CONFIGURATION ---
num_runs = 5
all_val_small = []
all_val_large = []

print(f"Starting {num_runs} independent runs...")

for run in range(num_runs):
    # Re-shuffle and Re-split every run for better averaging
    dataset = dataset.shuffle()
    
    # Split by size
    small_data = [d for d in dataset if d.num_nodes <= 40]
    large_data = [d for d in dataset if d.num_nodes > 40]

    # Split small_data into Train (80%) and Val_Small (20%)
    split = int(len(small_data) * 0.8)
    train_loader = DataLoader(small_data[:split], batch_size=64, shuffle=True)
    val_small_loader = DataLoader(small_data[split:], batch_size=64)
    val_large_loader = DataLoader(large_data, batch_size=64)

    model = RobustGNN(hidden_channels=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_small_acc = 0
    best_at_large_acc = 0
    patience = 20
    trigger_times = 0

    for epoch in range(1, 151):
        # Training
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        
        # Validation
        current_val_small = test(model, val_small_loader)
        
        if current_val_small > best_val_small_acc:
            best_val_small_acc = current_val_small
            # Check how we are doing on large graphs at our "best" small-graph checkpoint
            best_at_large_acc = test(model, val_large_loader)
            trigger_times = 0
        else:
            trigger_times += 1
        
        if trigger_times >= patience:
            break

    all_val_small.append(best_val_small_acc)
    all_val_large.append(best_at_large_acc)
    print(f"Run {run+1}: Small Acc = {best_val_small_acc:.4f}, Large Acc = {best_at_large_acc:.4f}")

# 3. FINAL AVERAGED RESULTS
print("\n" + "="*40)
print("FINAL AVERAGED RESULTS (5 ITERATIONS)")
print("="*40)
print(f"Validation (Small Graphs <= 40 nodes): {np.mean(all_val_small):.4f} ± {np.std(all_val_small):.4f}")
print(f"Validation (Large Graphs > 40 nodes):  {np.mean(all_val_large):.4f} ± {np.std(all_val_large):.4f}")
print(f"Generalization Gap: {abs(np.mean(all_val_small) - np.mean(all_val_large)):.4f}")
print("="*40)