import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool

# 1️⃣ Load dataset
dataset = TUDataset(root='data', name='Mutagenicity')
dataset = dataset.shuffle()
train_dataset, test_dataset = dataset[:3500], dataset[3500:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 2️⃣ GCN with variable layers
class GCN(torch.nn.Module):
    def __init__(self, num_layers=2, hidden_channels=128, pooling='max'):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dataset.num_node_features, hidden_channels))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # pooling choice
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Pooling must be 'mean', 'sum', or 'max'")
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.pool(x, batch)
        x = self.lin(x)
        return x

# 3️⃣ Training and testing
def train(model, loader, optimizer, criterion, epochs=5):
    model.train()
    for _ in range(epochs):
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

def test(model, loader, print_softmax=False):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(dim=1)
            correct += int((pred == data.y).sum())
            if print_softmax and i == 0:  # first batch only
                print("Softmax (first 5):", probs[:5])
                print("Predicted:", pred[:5], "| True:", data.y[:5])
    return correct / len(loader.dataset)

# 4️⃣ Hyperparameters
hidden_channels_list = [32, 64, 128]
num_layers_list = [3, 4, 5]
pooling_methods = ['mean', 'sum', 'max']
epochs = 5
lr = 0.01

# 5️⃣ Run all combinations
for pooling in pooling_methods:
    for hidden in hidden_channels_list:
        for layers in num_layers_list:
            print(f"\n--- Pool: {pooling} | Hidden: {hidden} | Layers: {layers} ---")
            model = GCN(num_layers=layers, hidden_channels=hidden, pooling=pooling)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()
            train(model, train_loader, optimizer, criterion, epochs=epochs)
            test_acc = test(model, test_loader, print_softmax=True)
            print(f"Final Test Acc: {test_acc:.4f}")