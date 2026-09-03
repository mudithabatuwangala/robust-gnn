import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class RobustGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(RobustGNN, self).__init__()
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.lin(x)

def run_diagnostic():
    dataset_path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step1.pt"
    model_path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\best_model.pt"

    print("="*85)
    print("                       DIAGNOSTIC BENCHMARK MANIFEST                 ")
    print("="*85)
    print(f"[*] EVALUATED SCRIPT    : test small on rest 3.py")
    print(f"[*] TRAINED MODEL USED  : {model_path}")
    print(f"[*] TRAINED ON          : Small ER Graph Configurations")
    print(f"[*] VALIDATED/TESTED ON : {dataset_path} (Partitioned explicitly by Size vs Type)")
    print(f"[*] MODEL POOLING LAYER : Global ADD Pooling")
    print("="*85 + "\n")

    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        print("Required baseline artifacts are missing.")
        return

    master_dataset = torch.load(dataset_path, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGNN(128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    cases = {
        "Small ER (Home Ground)": [],
        "Large ER (Size Stress)": [],
        "Small BA (Structure Stress)": [],
        "Large BA (Combined Stress)": []
    }

    for d in master_dataset:
        is_large = getattr(d, 'is_large', d.x.shape[0] > 40)
        g_type = getattr(d, 'graph_type', 'ER' if 'ER' in str(d) else 'BA')
        
        if not is_large and "ER" in g_type:
            cases["Small ER (Home Ground)"].append(d)
        elif is_large and "ER" in g_type:
            cases["Large ER (Size Stress)"].append(d)
        elif not is_large and "BA" in g_type:
            cases["Small BA (Structure Stress)"].append(d)
        elif is_large and "BA" in g_type:
            cases["Large BA (Combined Stress)"].append(d)

    NUM_TRIALS = 5
    results = {}

    print("--- Detailed Size & Structure Evaluation Breakdowns ---")
    for name, data_list in cases.items():
        if len(data_list) == 0:
            continue
        case_accs = []
        for _ in range(NUM_TRIALS):
            loader = DataLoader(data_list, batch_size=32, shuffle=True)
            correct = 0
            for data in loader:
                data = data.to(device)
                with torch.no_grad():
                    pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
                    correct += int((pred == data.y).sum())
            case_accs.append((correct / len(data_list)) * 100)
        
        mean_acc = np.mean(case_accs)
        results[name] = mean_acc
        print(f"-> Category: {name.ljust(28)} | Acc: {mean_acc:.2f}% (±{np.std(case_accs):.1f})")

    print("\n" + "="*85)
    print("SCIENTIFIC CONCLUSION REPORT:")
    gap_size = results["Small ER (Home Ground)"] - results["Large ER (Size Stress)"]
    gap_struct = results["Small ER (Home Ground)"] - results["Small BA (Structure Stress)"]

    if gap_size > 20:
        print(f"  [!] SIZE BIAS DETECTED: Performance dropped by {gap_size:.1f}% on Large ER variants.")
    else:
        print("  [✓] SIZE ROBUSTNESS VERIFIED: Model scales safely from Small to Large ER graphs.")

    if gap_struct > 20:
        print(f"  [!] STRUCTURAL BIAS DETECTED: Performance dropped by {gap_struct:.1f}% when switching to BA structures.")
    else:
        print("  [✓] STRUCTURAL ROBUSTNESS VERIFIED: Model handles structural transitions smoothly.")
    print("="*85)

if __name__ == "__main__":
    run_diagnostic()