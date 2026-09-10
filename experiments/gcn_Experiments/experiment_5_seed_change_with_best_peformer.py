import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

# -------------------------------
# 1. Reproducibility Function
# -------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Strict determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------
# 2. GCN Model Definition
# -------------------------------
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, num_layers=5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# -------------------------------
# 3. Training & Testing logic
# -------------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# -------------------------------
# 4. Main Execution Loop
# -------------------------------
seeds = [1, 42, 123, 999, 2024]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_accuracies = []

raw_dataset = TUDataset(root='data', name='Mutagenicity')

print(f"Starting training on device: {device}")

for seed in seeds:
    # 1. Set all seeds
    set_seed(seed)
    
    # 2. Manually shuffle the indices using the seed
    # This replaces dataset.shuffle(seed=seed)
    indices = torch.randperm(len(raw_dataset), generator=torch.Generator().manual_seed(seed))
    dataset = raw_dataset[indices]
    
    # 3. Split
    train_dataset = dataset[:3500]
    test_dataset = dataset[3500:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Initialize Model
    model = GCN(
        hidden_channels=128, 
        num_features=raw_dataset.num_node_features,
        num_classes=raw_dataset.num_classes,
        num_layers=5
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 5. Train
    for epoch in range(1, 11):
        train(model, train_loader, optimizer, criterion, device)
    
    # 6. Evaluate
    acc = test(model, test_loader, device)
    all_accuracies.append(acc)
    
    print(f"Seed: {seed:4d} | Accuracy: {acc:.4f}")

# -------------------------------
# 5. Final Stats
# -------------------------------
print("-" * 30)
print(f"Final Mean Accuracy: {np.mean(all_accuracies):.4f}")
print(f"Standard Deviation:  {np.std(all_accuracies):.4f}")