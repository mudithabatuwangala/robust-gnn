import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
from sklearn.model_selection import train_test_split

# 1. SETUP
dataset = TUDataset(root='/tmp/Mutagenicity', name='Mutagenicity').shuffle()
train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

train_loader = DataLoader(dataset[train_idx], batch_size=64, shuffle=True)
test_loader = DataLoader(dataset[test_idx], batch_size=64)

# 2. MODEL
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

model = RobustGNN(128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 3. FIXED TRAINING (100 Epochs)
print("Running Version 1: 80/20 Split, Fixed 100 Epochs")
for epoch in range(1, 101):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
    
    if epoch % 20 == 0:
        model.eval()
        correct = 0
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        print(f"Epoch {epoch:03d} | Test Acc: {correct/len(test_loader.dataset):.4f}")