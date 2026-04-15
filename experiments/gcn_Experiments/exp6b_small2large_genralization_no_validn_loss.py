import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
import torch.nn.functional as F

# 1. LOAD DATASET
dataset = TUDataset(root='/tmp/Mutagenicity', name='Mutagenicity').shuffle()

# 2. SPLIT DATA (Manual Size-Based Split)
# Train on Small/Medium only
train_data = [d for d in dataset if d.num_nodes <= 40]
# Test on Large only (The Challenge)
challenge_data = [d for d in dataset if d.num_nodes > 40]

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
challenge_loader = DataLoader(challenge_data, batch_size=64)

print(f"Training on {len(train_data)} Small graphs.")
print(f"Testing on {len(challenge_data)} Large graphs.")

# 3. DEFINE MODEL
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

model = RobustGNN(hidden_channels=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 4. TRAINING FUNCTION
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
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# 5. FIXED TRAINING LOOP (100 Epochs, No Early Stopping)
print("Starting training (Version 1.5: Fixed 100 Epochs)...")
for epoch in range(1, 101):
    loss = train()
    
    if epoch % 20 == 0:
        # We check the Challenge Accuracy periodically just to see the progress
        challenge_acc = test(challenge_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Challenge Acc: {challenge_acc:.4f}')

# 6. FINAL RESULTS
final_acc = test(challenge_loader)
print("-" * 30)
print(f'Final Challenge (Large Graphs) Accuracy: {final_acc:.4f}')