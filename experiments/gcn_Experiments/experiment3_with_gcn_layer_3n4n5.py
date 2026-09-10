import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool

# 1️⃣ Load dataset
dataset = TUDataset(root='data', name='Mutagenicity')
dataset = dataset.shuffle()
train_dataset, test_dataset = dataset[:3500], dataset[3500:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 2️⃣ GCN class with variable number of layers
class GCN(torch.nn.Module):
    def __init__(self, num_layers=2, hidden_channels=128):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dataset.num_node_features, hidden_channels))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x

# 3️⃣ Training and testing functions
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
            if print_softmax and i == 0:  # first batch
                print("Softmax probabilities (first 5 molecules):")
                print(probs[:5])
                print("Predicted:", pred[:5], "| True:", data.y[:5])
    return correct / len(loader.dataset)

# 4️⃣ Hyperparameters
hidden_channels = 128
epochs = 10
learning_rate = 0.01
layer_options = [3, 4, 5]  # test these depths

# 5️⃣ Run experiments for different layer depths
for num_layers in layer_options:
    print(f"\n=== GCN with {num_layers} layers ===")
    model = GCN(num_layers=num_layers, hidden_channels=hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        loss = train(model, train_loader, optimizer, criterion)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Train Acc {train_acc:.4f} | Test Acc {test_acc:.4f}")

    print("\nFinal Test Accuracy for first batch:")
    test(model, test_loader, print_softmax=True)