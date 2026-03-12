import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load Mutagenicity dataset
dataset = TUDataset(root='data', name='Mutagenicity')
dataset = dataset.shuffle()
train_dataset, test_dataset = dataset[:3500], dataset[3500:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# Basic info
print("Number of graphs (molecules):", len(dataset))
print("Number of node features per atom:", dataset.num_node_features)
print("Number of classes:", dataset.num_classes)

data = dataset[0]

print(data)
print("Node feature shape:", data.x.shape)
print("Edge index shape:", data.edge_index.shape)
print("Label:", data.y)


# 2. Visualize one molecule
data = dataset[0]
print(data.x)
G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(4,4))
nx.draw(G, node_size=50)
plt.title("Example Molecule Graph")
plt.show()


# 3. Define compact GCN
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=128):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 4. Training
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 5. Testing with softmax
def test(loader, print_batch=False):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(dim=1)
            correct += int((pred == data.y).sum())
            
            if print_batch:  # print only first batch
                print("Softmax probabilities (first 5 molecules):")
                print(probs[:5])
                print("Predicted:", pred[:5], "| True:", data.y[:5])
                break
    return correct / len(loader.dataset)

# 6. Training loop
for epoch in range(1, 11):  # reduced to 10 epochs for report screenshot
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Train Acc {train_acc:.4f} | Test Acc {test_acc:.4f}")

# 7. Final Test Accuracy with softmax print
print("\nFinal Test Accuracy:")
test(test_loader, print_batch=True)