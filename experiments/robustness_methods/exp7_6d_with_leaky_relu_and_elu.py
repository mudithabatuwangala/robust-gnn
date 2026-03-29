import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_max_pool
from sklearn.model_selection import train_test_split
import numpy as np
import copy

# 1. LOAD DATASET
dataset = TUDataset(root='/tmp/Mutagenicity', name='Mutagenicity')

# 2. STRATIFIED SPLIT (Size + Label)
categories = []
for d in dataset:
    size_tag = "L" if d.num_nodes > 40 else "S"
    label_tag = str(d.y.item())
    categories.append(size_tag + label_tag)

indices = np.arange(len(dataset))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=categories, random_state=42)
temp_cats = [categories[i] for i in temp_idx]
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_cats, random_state=42)

train_loader = DataLoader(dataset[train_idx.tolist()], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[val_idx.tolist()], batch_size=64)
test_loader = DataLoader(dataset[test_idx.tolist()], batch_size=64)

# 3. DEFINE MODEL WITH CUSTOM ACTIVATION
class RobustGNN(torch.nn.Module):
    def __init__(self, hidden_channels, activation_type='elu'):
        super(RobustGNN, self).__init__()
        torch.manual_seed(42)
        self.activation_type = activation_type
        self.conv1 = GATConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, hidden_channels)
        self.conv5 = GATConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Select Activation
        if self.activation_type == 'elu':
            act = F.elu
        else:
            act = F.leaky_relu

        x = act(self.conv1(x, edge_index))
        x = act(self.conv2(x, edge_index))
        x = act(self.conv3(x, edge_index))
        x = act(self.conv4(x, edge_index))
        x = act(self.conv5(x, edge_index))
        
        x = global_max_pool(x, batch)
        return self.lin(x)

# CHOOSE YOUR TEST HERE: 'elu' or 'leaky_relu'
CURRENT_ACT = 'elu' 

model = RobustGNN(hidden_channels=128, activation_type=CURRENT_ACT)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 4. UTILITY FUNCTIONS
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
        val_loss += criterion(out, data.y).item() * data.num_graphs
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

# 5. TRAINING LOOP
best_val_loss = float('inf')
best_model_state = None
patience = 20
trigger_times = 0

print(f"Starting training with Activation: {CURRENT_ACT.upper()}...")
for epoch in range(1, 151):
    train_loss = train()
    val_loss = get_val_loss()
    
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
        print(f"Early stopping at epoch {epoch}")
        break

# 6. FINAL BREAKDOWN
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
print(f"RESULTS FOR ACTIVATION: {CURRENT_ACT.upper()}")
print(f'Best Val Loss achieved: {best_val_loss:.4f}')
print(f'Test Accuracy (Small <= 40): {small_correct/small_total:.4f}')
print(f'Test Accuracy (Large > 40): {large_correct/large_total:.4f}')