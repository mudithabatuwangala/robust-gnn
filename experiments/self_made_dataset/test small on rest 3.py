import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
import os
import numpy as np


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
    # Path configuration
    dataset_path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step1.pt"
    
    # Load Model (Ensure you have saved the model state during training)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGNN(hidden_channels=128).to(device)
    
    # If not, you might need to re-run training once with a save command.
    model_path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\best_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} nahi mila. Training script mein model save karein.")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load Dataset
    dataset = torch.load(dataset_path, weights_only=False)

    # 2. BUCKETING LOGIC (The 4 Cases)
    cases = {
        "Case A: Small ER (Home Ground)": [d for d in dataset if d.graph_type == "ER_Small"],
        "Case B: Large ER (Size Stress)": [d for d in dataset if d.graph_type == "ER_Large"],
        "Case C: Small BA (Structure Stress)": [d for d in dataset if d.graph_type == "BA_Small"],
        "Case D: Large BA (Total Stress)": [d for d in dataset if d.graph_type == "BA_Large"]
    }

    print("\n" + "="*60)
    print("         GNN DIAGNOSTIC REPORT: STEP 2")
    print("="*60)
    print(f"{'Test Case'.ljust(35)} | {'Samples'.ljust(8)} | {'Accuracy'}")
    print("-" * 60)

    results = {}
    for name, data_list in cases.items():
        if len(data_list) == 0:
            print(f"{name.ljust(35)} | {'0'.ljust(8)} | N/A")
            continue
            
        loader = DataLoader(data_list, batch_size=32)
        correct = 0
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
                correct += int((pred == data.y).sum())
        
        acc = (correct / len(data_list)) * 100
        results[name] = acc
        print(f"{name.ljust(35)} | {str(len(data_list)).ljust(8)} | {acc:.2f}%")

    print("="*60)

    # 3. AUTOMATED REASONING
    print("\nSCIENTIFIC CONCLUSION:")
    
    gap_size = results["Case A: Small ER (Home Ground)"] - results["Case B: Large ER (Size Stress)"]
    gap_struct = results["Case A: Small ER (Home Ground)"] - results["Case C: Small BA (Structure Stress)"]

    if gap_size > 20:
        print(f"-> SIZE BIAS DETECTED: Accuracy dropped by {gap_size:.1f}% on Large ER.")
        print("   Reason: 5 layers are insufficient to pass messages in 100+ nodes (Over-smoothing/Dilution).")
    
    if gap_struct > 20:
        print(f"-> STRUCTURAL BIAS DETECTED: Accuracy dropped by {gap_struct:.1f}% on Small BA.")
        print("   Reason: Model is overfitted to uniform degrees and cannot handle 'Hubs'.")

    if results["Case D: Large BA (Total Stress)"] < 55:
        print("-> TOTAL GENERALIZATION FAILURE: Combined Size + Structure effect makes the model blind.")

if __name__ == "__main__":
    run_diagnostic()