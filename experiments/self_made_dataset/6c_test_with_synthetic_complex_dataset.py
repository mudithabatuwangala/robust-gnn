import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
import copy
import os

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

def run_complex_experiment():
    dataset_path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step2_complex.pt"
    model_save_path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\best_complex_model.pt"

    if not os.path.exists(dataset_path):
        print("Error: Complex dataset nahi mila! Pehle generator chalayein.")
        return

    full_dataset = torch.load(dataset_path, weights_only=False)

    # STEP 2 SETUP: Train on Small ER (Complex Logic)
    train_val_data = [d for d in full_dataset if d.graph_type == "ER_Small"]
    split = int(len(train_val_data) * 0.8)
    train_loader = DataLoader(train_val_data[:split], batch_size=32, shuffle=True)
    val_loader = DataLoader(train_val_data[split:], batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGNN(128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    patience = 15
    trigger = 0
    best_model_state = None

    print("\nTraining on Complex Logic (Small ER)...")
    for epoch in range(1, 101):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        for data in val_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
            correct += int((pred == data.y).sum())
        val_acc = correct / len(val_loader.dataset)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            trigger = 0
        else:
            trigger += 1
        
        if epoch % 10 == 0 or trigger == 0:
            print(f"Epoch {epoch:03d} | Val Acc: {val_acc:.4f} | Trigger: {trigger}")
        
        if trigger >= patience: break

    # SAVE AND DIAGNOSE
    torch.save(best_model_state, model_save_path)
    model.load_state_dict(best_model_state)
    
    print("\n" + "="*60)
    print("      DIAGNOSTIC REPORT: COMPLEX MOLECULAR LOGIC")
    print("="*60)
    
    cases = {
        "Case A: Small ER (Home)": [d for d in full_dataset if d.graph_type == "ER_Small"],
        "Case B: Large ER (Size)": [d for d in full_dataset if d.graph_type == "ER_Large"],
        "Case C: Small BA (Struct)": [d for d in full_dataset if d.graph_type == "BA_Small"],
        "Case D: Large BA (Total)": [d for d in full_dataset if d.graph_type == "BA_Large"]
    }

    for name, data_list in cases.items():
        loader = DataLoader(data_list, batch_size=32)
        correct = 0
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
                correct += int((pred == data.y).sum())
        acc = (correct / len(data_list)) * 100
        print(f"{name.ljust(30)} | Accuracy: {acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_complex_experiment()