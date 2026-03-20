# import torch
# import torch.nn.functional as F
# from torch.nn import Linear
# from torch_geometric.datasets import TUDataset
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GCNConv, global_mean_pool

# # --- 1. DATA PREPARATION (SIZE-BASED) ---
# def prepare_size_datasets():
#     full_dataset = TUDataset(root='./data', name='Mutagenicity')
    
#     # Filter for Small Graphs (< 25 nodes)
#     small_graphs = [data for data in full_dataset if data.num_nodes < 25]
#     # Filter for Large Graphs (> 40 nodes)
#     large_graphs = [data for data in full_dataset if data.num_nodes > 40]
    
#     # Split Small Graphs: 80% for Training, 20% for testing small accuracy
#     torch.manual_seed(123)
#     indices = torch.randperm(len(small_graphs))
#     split = int(len(small_graphs) * 0.8)
    
#     train_data = [small_graphs[i] for i in indices[:split]]
#     test_small_data = [small_graphs[i] for i in indices[split:]]
    
#     print(f"Total Mutagenicity Dataset: {len(full_dataset)}")
#     print(f"Training on: {len(train_data)} Small Graphs (<25 nodes)")
#     print(f"Testing on: {len(test_small_data)} Small Graphs (Check)")
#     print(f"Testing on: {len(large_graphs)} Large Graphs (>40 nodes - STRESS TEST)")
    
#     return train_data, test_small_data, large_graphs, full_dataset.num_node_features, full_dataset.num_classes

# # --- 2. THE GCN MODEL ---
# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels, num_features, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)
#         self.lin = Linear(hidden_channels, num_classes)

#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index).relu()
#         x = self.conv3(x, edge_index)
#         x = global_mean_pool(x, batch) 
#         x = F.dropout(x, p=0.5, training=self.training)
#         return self.lin(x)

# # --- 3. HELPER FUNCTIONS ---
# def train(model, loader, optimizer, criterion):
#     model.train()
#     for data in loader:
#         out = model(data.x, data.edge_index, data.batch)
#         loss = criterion(out, data.y)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

# def evaluate(model, loader):
#     model.eval()
#     correct = 0
#     for data in loader:
#         out = model(data.x, data.edge_index, data.batch)
#         pred = out.argmax(dim=1)
#         correct += int((pred == data.y).sum())
#     return correct / len(loader.dataset)

# # --- 4. RUNNING THE EXPERIMENT ---
# train_data, test_small, test_large, n_feats, n_classes = prepare_size_datasets()

# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# small_loader = DataLoader(test_small, batch_size=32, shuffle=False)
# large_loader = DataLoader(test_large, batch_size=32, shuffle=False)

# model = GCN(64, n_feats, n_classes)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()

# print("\nStarting Training...")
# for epoch in range(1, 101):
#     train(model, train_loader, optimizer, criterion)
#     if epoch % 20 == 0:
#         s_acc = evaluate(model, small_loader)
#         l_acc = evaluate(model, large_loader)
#         print(f"Epoch {epoch:03d} | Small Acc: {s_acc:.4f} | Large Acc (Stress): {l_acc:.4f}")


import torch
from torch_geometric.loader import DataLoader
from gcn import GCN  # Import the brain
from utils import prepare_size_datasets, train, evaluate  # Import the tools

# 1. Setup Data
train_data, test_small, test_large, n_feats, n_classes = prepare_size_datasets()

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
small_loader = DataLoader(test_small, batch_size=32, shuffle=False)
large_loader = DataLoader(test_large, batch_size=32, shuffle=False)

# 2. Setup Model
model = GCN(hidden_channels=64, num_features=n_feats, num_classes=n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 3. The Loop
print("Training on small graphs...")
for epoch in range(1, 101):
    train(model, train_loader, optimizer, criterion)
    if epoch % 20 == 0:
        s_acc = evaluate(model, small_loader)
        l_acc = evaluate(model, large_loader)
        print(f"Epoch {epoch:03d} | Small Acc: {s_acc:.4f} | Large Acc (STRESS): {l_acc:.4f}")