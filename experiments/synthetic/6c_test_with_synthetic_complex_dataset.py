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


# clean evaluation function 
def evaluate(loader, model, device):
    all_preds = []
    all_targets = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(data.y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = (np.sum(all_preds == all_targets) / len(all_targets)) * 100

    p, r, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0
    )

    return acc, p * 100, r * 100, f1 * 100


def run_complex_experiment():
    dataset_path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step2_complex.pt"
    model_path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\best_complex_model.pt"
    
    print("="*70)
    print("                   EVALUATION MANIFEST & REPORT              ")
    print("="*70)
    print(f"[*] TRAINED MODEL USED  : {model_path}")
    print(f"[*] TRAINED ON          : Synthetic complex datasets (Molecular logic)")
    print(f"[*] VALIDATED/TESTED ON : {dataset_path} (Sparsity & Connectivity subsets)")
    print(f"[*] MODEL POOLING LAYER : Global ADD Pooling")
    print("="*70 + "\n")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    full_dataset = torch.load(dataset_path, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RobustGNN(128).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Size-based split (validation conditions)
    cases = {
        "Small Graphs (<=35 nodes)": [d for d in full_dataset if d.x.shape[0] <= 35],
        "Large Graphs (>35 nodes)": [d for d in full_dataset if d.x.shape[0] > 35]
    }

    NUM_TRIALS = 5
    
    for name, data_list in cases.items():
        if len(data_list) == 0:
            print(f"[{name}] No samples found.\n")
            continue
            
        accs, precs, recs, f1s = [], [], [], []
        
        for _ in range(NUM_TRIALS):
            loader = DataLoader(data_list, batch_size=32, shuffle=True)

            acc, p, r, f1 = evaluate(loader, model, device)

            accs.append(acc)
            precs.append(p)
            recs.append(r)
            f1s.append(f1)
            
        print(f"{name.ljust(30)} | "
              f"Acc: {np.mean(accs):.2f}% (±{np.std(accs):.2f}) | "
              f"P: {np.mean(precs):.2f}% | "
              f"R: {np.mean(recs):.2f}% | "
              f"F1: {np.mean(f1s):.2f}%")

    print("="*70)


if __name__ == "__main__":
    run_complex_experiment()