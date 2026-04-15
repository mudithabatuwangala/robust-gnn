import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load Mutagenicity dataset
dataset = TUDataset(root='data', name='Mutagenicity')
dataset = dataset.shuffle()
train_dataset, test_dataset = dataset[:3500], dataset[3500:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 2. Visualize one molecule (always same graph for report)
torch.manual_seed(42)
G = to_networkx(dataset[0], to_undirected=True)
plt.figure(figsize=(4,4))
nx.draw(G, node_size=50)
plt.title("Example Molecule Graph")
plt.show()

# 3. Define GCN with switchable pooling
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=128, pooling='mean'):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        # choose pooling
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Pooling must be 'mean', 'sum', or 'max'")

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        x = self.lin(x)
        return x

# 4. Training and testing functions
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, print_softmax=False):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(dim=1)
            correct += int((pred == data.y).sum())
            if print_softmax and i == 0:  # print only first batch
                print("Softmax probabilities (first 5 molecules):")
                print(probs[:5])
                print("Predicted:", pred[:5], "| True:", data.y[:5])
    return correct / len(loader.dataset)

# 5. Hyperparameters
hidden_channels = 128
pooling_methods = ['mean', 'sum', 'max']
lr = 0.01
epochs = 10

# 6. Run experiments for each pooling method
for pooling in pooling_methods:
    print(f"\n=== Pooling Method: {pooling} ===")
    model = GCN(hidden_channels=hidden_channels, pooling=pooling)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs+1):
        loss = train(model, train_loader, optimizer, criterion)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Train Acc {train_acc:.4f} | Test Acc {test_acc:.4f}")
    
    print("\nFinal Test Accuracy:")
    test(model, test_loader, print_softmax=True)