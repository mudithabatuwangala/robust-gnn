import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
from sklearn.model_selection import train_test_split
import numpy as np
import copy

# 1. LOAD DATASET
dataset = TUDataset(root='/tmp/Mutagenicity', name='Mutagenicity')

# 2. STRATIFIED SPLIT (Size + Label)
# We create a combined tag: e.g., 'S1' (Small/Toxic) or 'L0' (Large/Safe)
categories = []
for d in dataset:
    size_tag = "L" if d.num_nodes > 40 else "S"
    label_tag = str(d.y.item())
    categories.append(size_tag + label_tag)

# Split 1: 70% Train, 30% Temp
indices = np.arange(len(dataset))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, stratify=categories, random_state=42
)

# Split 2: Divide Temp into 50% Val and 50% Test (15% each of total)
temp_cats = [categories[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=temp_cats, random_state=42
)

train_loader = DataLoader(dataset[train_idx.tolist()], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[val_idx.tolist()], batch_size=64)
test_loader = DataLoader(dataset[test_idx.tolist()], batch_size=64)

print(f"Train size: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

# 3. DEFINE MODEL
class RobustGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(RobustGNN, self).__init__()
        torch.manual_seed(42) # Seed weights for Version 3 rigor
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

model = RobustGNN(hidden_channels=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 4. TRAINING & VALIDATION LOSS FUNCTIONS
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def get_val_loss():
    model.eval()
    val_loss = 0
    for data in val_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        val_loss += loss.item() * data.num_graphs
    return val_loss / len(val_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# 5. TRAINING LOOP (Early Stopping on Val Loss)
best_val_loss = float('inf')
best_model_state = None
patience = 20
trigger_times = 0

print("Starting training Version 3...")
for epoch in range(1, 151):
    train_loss = train()
    val_loss = get_val_loss()
    
    # In Version 3, we look for LOWER loss, not higher accuracy
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict()) 
        trigger_times = 0
    else:
        trigger_times += 1
    
    if epoch % 10 == 0:
        val_acc = test(val_loader)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    if trigger_times >= patience:
        print(f"Early stopping at epoch {epoch} (Val Loss stopped improving)")
        break

# 6. FINAL BREAKDOWN (The "Why")
model.load_state_dict(best_model_state)
model.eval()

small_correct, small_total = 0, 0
large_correct, large_total = 0, 0

with torch.no_grad():
    for data in test_loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        for i in range(len(data.y)):
            nodes = (data.batch == i).sum().item()
            is_correct = (pred[i] == data.y[i]).item()
            if nodes <= 40:
                small_correct += is_correct
                small_total += 1
            else:
                large_correct += is_correct
                large_total += 1

print("-" * 30)
print(f'Best Val Loss achieved: {best_val_loss:.4f}')
print(f'Test Accuracy (Small <= 40): {small_correct/small_total:.4f}')
print(f'Test Accuracy (Large > 40): {large_correct/large_total:.4f}')