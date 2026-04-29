import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from sklearn.model_selection import train_test_split

# FIX for PyTorch 2.6 weights_only loading
if not hasattr(torch, "_orig_load"):
    torch._orig_load = torch.load
torch.load = lambda *args, **kwargs: torch._orig_load(*args, **{**kwargs, "weights_only": False})

# 1. Load and Stratify Synthetic Data
def prepare_synthetic_data():
    print("Loading Synthetic Scale Dataset...")
    # Load the dataset you just generated
    dataset = torch.load('hiv_synthetic_scale_data.pt')
    
    for d in dataset:
        d.x = d.x.float()
        d.y = d.y.view(-1).long()

    # We split based on node count to ensure "Small-Synthetic" and "Large-Synthetic" 
    # are present in both Train and Val sets.
    # Using 120 nodes as the threshold based on your 26-293 range.
    small_synth = [d for d in dataset if d.num_nodes < 120]
    large_synth = [d for d in dataset if d.num_nodes >= 120]

    s_train, s_val = train_test_split(small_synth, train_size=0.8, random_state=42)
    l_train, l_val = train_test_split(large_synth, train_size=0.8, random_state=42)

    train_set = s_train + l_train
    val_set = s_val + l_val

    print(f"Split Summary: Train={len(train_set)} | Val={len(val_set)}")
    return train_set, val_set, dataset[0].num_node_features

# 2. Setup Loaders
train_data, val_data, feat_dim = prepare_synthetic_data()

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# We keep them in one loader but will filter during evaluation for accuracy reporting
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 3. Model (Optimized GCN with SUM Pooling)
class SyntheticGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SyntheticGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        # SUM pooling to capture total signal from composed structures
        x = global_add_pool(x, batch) 
        return self.lin(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SyntheticGCN(feat_dim, 128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 4. Evaluation Function (Separated by Size)
def evaluate_by_size(loader):
    model.eval()
    # Metrics for "Lower-Large" (<120) and "Higher-Large" (>=120)
    low_correct, low_total = 0, 0
    high_correct, high_total = 0, 0
    
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            
            for i in range(len(data.y)):
                # Calculate size of individual graph in batch
                num_nodes = (data.batch == i).sum().item()
                correct = int(pred[i] == data.y[i])
                
                if num_nodes < 120:
                    low_correct += correct
                    low_total += 1
                else:
                    high_correct += correct
                    high_total += 1
                    
    acc_low = low_correct / low_total if low_total > 0 else 0
    acc_high = high_correct / high_total if high_total > 0 else 0
    return acc_low, acc_high

# 5. Training Loop
print("\nTraining on Synthetic Composed Graphs...")
best_acc = 0

for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Evaluate every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        acc_low, acc_high = evaluate_by_size(val_loader)
        avg_val = (acc_low + acc_high) / 2
        
        if avg_val > best_acc:
            best_acc = avg_val
            torch.save(model.state_dict(), 'synthetic_scale_model.pth')
            
        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Acc <120 nodes: {acc_low:.4f} | Acc >=120 nodes: {acc_high:.4f}")

print(f"\nFinal Best Model Saved. Best Average Val Acc: {best_acc:.4f}")