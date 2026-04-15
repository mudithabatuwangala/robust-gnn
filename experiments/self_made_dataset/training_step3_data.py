import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
import os

# --- BASE 6c ARCHITECTURE (Standard) ---
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
        x = global_max_pool(x, batch)
        return self.lin(x)

def train_model():
    dataset_path = "step3_6c_data_small_complex.pt"
    model_save_path = "step3_trained_model.pt"

    if not os.path.exists(dataset_path):
        print("Error: Dataset not found")
        return

    full_dataset = torch.load(dataset_path, weights_only=False)
    
    # Simple Shuffle and Split
    torch.manual_seed(42)
    split = int(len(full_dataset) * 0.8)
    train_loader = DataLoader(full_dataset[:split], batch_size=32, shuffle=True)
    val_loader = DataLoader(full_dataset[split:], batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGNN(128).to(device)
    
    # Standard Optimizer (No Weight Decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    print("\n--- Training on New Complex Small Dataset ---")
    for epoch in range(1, 101): # Standard 100 epochs
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
                correct += int((pred == data.y).sum())
        
        val_acc = correct / len(val_loader.dataset)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Val Acc: {val_acc:.4f}")

    print(f"Training Complete. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_model()