import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

# Load dataset
dataset = TUDataset(root='data', name='Mutagenicity')
dataset = dataset.shuffle()
train_dataset = dataset[:3500]
test_dataset = dataset[3500:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Pooling methods to test
poolings = {
    'mean': global_mean_pool,
    'sum': global_add_pool,
    'max': global_max_pool
}

# Seeds to test
seeds = [1, 42, 123, 999, 2024]

# Hidden channels and layers to test
hidden_list = [32, 64, 128]
layer_list = [3, 4, 5]

# Experiment loop
for pool_name, pool_fn in poolings.items():
    for hidden in hidden_list:
        for layers in layer_list:
            for seed in seeds:
                torch.manual_seed(seed)
                
                # Define dynamic GCN
                class GCN(torch.nn.Module):
                    def __init__(self, hidden_channels, num_layers):
                        super().__init__()
                        self.convs = torch.nn.ModuleList()
                        self.convs.append(GCNConv(dataset.num_node_features, hidden_channels))
                        for _ in range(num_layers-1):
                            self.convs.append(GCNConv(hidden_channels, hidden_channels))
                        self.lin = Linear(hidden_channels, dataset.num_classes)

                    def forward(self, x, edge_index, batch):
                        for conv in self.convs:
                            x = F.relu(conv(x, edge_index))
                        x = pool_fn(x, batch)
                        x = self.lin(x)
                        return x

                # Initialize model
                model = GCN(hidden_channels=hidden, num_layers=layers)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                criterion = torch.nn.CrossEntropyLoss()

                # Training
                model.train()
                for epoch in range(10):  # keep epochs small for quick testing
                    for data in train_loader:
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index, data.batch)
                        loss = criterion(out, data.y)
                        loss.backward()
                        optimizer.step()

                # Testing
                model.eval()
                correct = 0
                first_batch_softmax = None
                first_batch_pred = None
                first_batch_true = None
                for i, data in enumerate(test_loader):
                    out = model(data.x, data.edge_index, data.batch)
                    pred = out.argmax(dim=1)
                    correct += int((pred == data.y).sum())
                    if i == 0:
                        first_batch_softmax = F.softmax(out, dim=1)
                        first_batch_pred = pred
                        first_batch_true = data.y
                acc = correct / len(test_loader.dataset)

                # Print concise result
                print(f"\nPool: {pool_name} | Hidden: {hidden} | Layers: {layers} | Seed: {seed}")
                print(f"Test Acc: {acc:.4f}")
                print("Softmax (first 5):")
                print(first_batch_softmax[:5])
                print("Predicted:", first_batch_pred[:5].tolist(), "| True:", first_batch_true[:5].tolist())