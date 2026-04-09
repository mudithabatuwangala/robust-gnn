import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_max_pool
import torch.nn as nn
import copy

dataset = TUDataset(root='/tmp/Mutagenicity', name='Mutagenicity').shuffle()

train_val_data = [d for d in dataset if d.num_nodes <= 40]
challenge_data = [d for d in dataset if d.num_nodes > 40]

split = int(len(train_val_data) * 0.8)
train_loader = DataLoader(train_val_data[:split], batch_size=64, shuffle=True)
val_loader = DataLoader(train_val_data[split:], batch_size=64)
challenge_loader = DataLoader(challenge_data, batch_size=64)

print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Challenge size: {len(challenge_data)}")

class RobustGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        def mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ELU(),
                nn.Linear(out_dim, out_dim)
            )

        self.conv1 = GINConv(mlp(dataset.num_node_features, hidden_channels))
        self.conv2 = GINConv(mlp(hidden_channels, hidden_channels))
        self.conv3 = GINConv(mlp(hidden_channels, hidden_channels))
        self.conv4 = GINConv(mlp(hidden_channels, hidden_channels))
        self.conv5 = GINConv(mlp(hidden_channels, hidden_channels))

        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = F.elu(self.conv5(x, edge_index))
        x = global_max_pool(x, batch)
        return self.lin(x)

model = RobustGNN(128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

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

best_val_acc, patience, trigger_times = 0, 20, 0
best_model_state = None

print("Starting training Version ELU Max...")
for epoch in range(1, 151):
    loss = train()
    val_acc = test(val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        trigger_times = 0
    else:
        trigger_times += 1

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

    if trigger_times >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_model_state)
challenge_acc = test(challenge_loader)

print("-"*30)
print(f'Final Best Val Acc (Small/Med): {best_val_acc:.4f}')
print(f'Final Challenge (Large Graphs) Acc: {challenge_acc:.4f}')