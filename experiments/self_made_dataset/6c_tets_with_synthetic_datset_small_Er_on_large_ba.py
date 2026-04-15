import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn.functional as F
import copy
import os

# 1. MODEL DEFINITION (6c Architecture)
class RobustGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(RobustGNN, self).__init__()
        # 5 Layers for deep message passing
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
        
        # Global Pooling to get graph-level representation
        x = global_add_pool(x, batch)
        return self.lin(x)

def train_step1():
    path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step1.pt"
    
    if not os.path.exists(path):
        print("Error: Dataset file nahi mili! Pehle generator chalayein.")
        return

    # Load full dataset with PyTorch 2.6 fix
    full_dataset = torch.load(path, weights_only=False)

    # 2. FILTERING FOR STEP 1
    # Train/Val: Only Small Erdős-Rényi
    train_val_data = [d for d in full_dataset if d.graph_type == "ER_Small"]
    # Challenge: Only Large Barabási-Albert
    challenge_data = [d for d in full_dataset if d.graph_type == "BA_Large"]

    print(f"Step 1 Setup:")
    print(f"  - Training/Val Samples (Small ER): {len(train_val_data)}")
    print(f"  - Challenge Samples (Large BA): {len(challenge_data)}")

    # Split Train/Val (80/20)
    split = int(len(train_val_data) * 0.8)
    train_loader = DataLoader(train_val_data[:split], batch_size=32, shuffle=True)
    val_loader = DataLoader(train_val_data[split:], batch_size=32)
    challenge_loader = DataLoader(challenge_data, batch_size=32)

    # 3. TRAINING SETUP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGNN(hidden_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Early Stopping Variables
    best_val_acc = 0
    patience = 15
    trigger = 0
    best_model_state = None

    print("\nStarting Training...")
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

        # Validation
        model.eval()
        correct = 0
        for data in val_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
            correct += int((pred == data.y).sum())
        val_acc = correct / len(val_loader.dataset)

        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            trigger = 0
        else:
            trigger += 1

        if epoch % 10 == 0 or trigger == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if trigger >= patience:
            print(f"Early Stopping triggered at epoch {epoch}")
            break

    
   # 4. FINAL CHALLENGE REPORT
    print("\n" + "="*40)
    print("         STEP 1 FINAL REPORT")
    print("="*40)
    
    model.load_state_dict(best_model_state)
    model.eval()
    
    correct = 0
    for data in challenge_loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
        correct += int((pred == data.y).sum())
    
    challenge_acc = correct / len(challenge_loader.dataset)
    
    print(f"Small ER (Training) Best Val Acc : {best_val_acc*100:.2f}%")
    print(f"Large BA (Challenge) Accuracy    : {challenge_acc*100:.2f}%")
    print(f"Generalization Gap               : {(best_val_acc - challenge_acc)*100:.2f}%")
    print("="*40)

    # --- YAHAN PATH UPDATE KIYA HAI ---
    model_save_path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\best_model.pt"
    torch.save(best_model_state, model_save_path)
    print(f"Model saved successfully at: {model_save_path}")

if __name__ == "__main__":
    train_step1()