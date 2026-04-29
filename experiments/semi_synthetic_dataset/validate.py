import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data

# FIX for PyTorch 2.6 weights_only loading
if not hasattr(torch, "_orig_load"):
    torch._orig_load = torch.load
torch.load = lambda *args, **kwargs: torch._orig_load(*args, **{**kwargs, "weights_only": False})

# 1. Model Architecture (Must match the trainer exactly)
class SyntheticGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SyntheticGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x.float(), edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_add_pool(x, batch) 
        return self.lin(x)

# 2. Load the Real HIV Dataset and Split by Size
def load_real_hiv():
    print("Loading Original HIV Dataset...")
    dataset = MoleculeNet(root='data/HIV', name='HIV')
    
    real_small = []
    real_large = []

    for d in dataset:
        # Standardize data for the model
        clean_d = Data(x=d.x, edge_index=d.edge_index, y=d.y.view(-1).long())
        
        if d.num_nodes <= 40:
            real_small.append(clean_d)
        elif d.num_nodes >= 100:
            real_large.append(clean_d)
            
    print(f"Found {len(real_small)} Real Small molecules and {len(real_large)} Real Large molecules.")
    return real_small, real_large, dataset.num_node_features

# 3. Validation Logic
def validate_on_real(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
    return correct / total if total > 0 else 0

# --- Execution ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
small_list, large_list, feat_dim = load_real_hiv()
small_loader = DataLoader(small_list, batch_size=32)
large_loader = DataLoader(large_list, batch_size=32)

# Load Saved Model
model = SyntheticGCN(feat_dim, 128).to(device)
try:
    model.load_state_dict(torch.load('synthetic_scale_model.pth', map_location=device))
    print("Successfully loaded 'synthetic_scale_model.pth'")
except FileNotFoundError:
    print("Error: Could not find the saved model file.")
    exit()

# Run Evaluation
print("\n--- Testing Synthetic Model on REAL Data ---")
small_acc = validate_on_real(model, small_loader, device)
large_acc = validate_on_real(model, large_loader, device)

print(f"Accuracy on Real SMALL Molecules (<= 40 nodes): {small_acc:.4f}")
print(f"Accuracy on Real LARGE Molecules (>= 100 nodes): {large_acc:.4f}")

# Analysis
if large_acc > small_acc:
    print("\nResult: The model generalizes BETTER to large molecules!")
else:
    print("\nResult: Performance is higher on small molecules.")