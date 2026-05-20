import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
import copy
import random

# =========================
# CONFIG
# =========================
TASK_IDX = 10
NUM_RUNS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# LOAD DATASET
# =========================
dataset = MoleculeNet(root='/tmp/Tox21', name='Tox21')

# =========================
# FILTER DATA
# =========================
filtered_dataset = []
for data in dataset:
    y = data.y[:, TASK_IDX]

    if not torch.isnan(y):
        data.y = y.long()
        data.x = data.x.float()
        filtered_dataset.append(data)

print(f"Filtered dataset size: {len(filtered_dataset)}")

# =========================
# MODEL
# =========================
class RobustGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = global_max_pool(x, batch)
        return self.lin(x).view(-1)

# =========================
# TRAIN FUNCTION
# =========================
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

# =========================
# TEST FUNCTION
# =========================
@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        pred = (torch.sigmoid(out) > 0.5).long()

        correct += int((pred == data.y).sum())
        total += data.y.size(0)

    return correct / total if total > 0 else 0

# =========================
# COUNT CLASSES FUNCTION
# =========================
def count_classes(data_list):
    toxic = 0
    non_toxic = 0

    for d in data_list:
        if int(d.y.item()) == 1:
            toxic += 1
        else:
            non_toxic += 1

    return toxic, non_toxic

# =========================
# EXPERIMENT LOOP
# =========================
small_accs = []
large_accs = []

train_sizes = []
val_sizes = []
test_sizes = []

for run in range(NUM_RUNS):
    print(f"\n===== RUN {run+1} =====")

    # -------- Split by size --------
    small_graphs = [d for d in filtered_dataset if d.num_nodes <= 40]
    large_graphs = [d for d in filtered_dataset if d.num_nodes > 40]

    random.shuffle(small_graphs)
    random.shuffle(large_graphs)

    # -------- SMALL: 60 / 20 / 20 --------
    n_small = len(small_graphs)

    train_small = small_graphs[:int(0.6 * n_small)]
    val_small   = small_graphs[int(0.6 * n_small):int(0.8 * n_small)]
    test_small  = small_graphs[int(0.8 * n_small):]

    # -------- LARGE: 0 / 20 / 80 --------
    n_large = len(large_graphs)

    val_large  = large_graphs[:int(0.2 * n_large)]
    test_large = large_graphs[int(0.2 * n_large):]

    val_data = val_small + val_large
    test_all = test_small + test_large

    # -------- PRINT SPLITS --------
    print("\n--- Data Split ---")
    print(f"Train (Small): {len(train_small)}")

    print(f"Validation (Small): {len(val_small)}")
    print(f"Validation (Large): {len(val_large)}")
    print(f"Total Validation: {len(val_small) + len(val_large)}")

    print(f"Test (Small): {len(test_small)}")
    print(f"Test (Large): {len(test_large)}")
    print(f"Total Test: {len(test_small) + len(test_large)}")

    # -------- CLASS DISTRIBUTION --------
    toxic, non_toxic = count_classes(test_all)

    print("\n--- Test Set Class Distribution ---")
    print(f"Toxic (1): {toxic}")
    print(f"Non-Toxic (0): {non_toxic}")
    print(f"Total: {toxic + non_toxic}")

    # Store sizes
    train_sizes.append(len(train_small))
    val_sizes.append(len(val_small) + len(val_large))
    test_sizes.append(len(test_small) + len(test_large))

    # -------- LOADERS --------
    train_loader = DataLoader(train_small, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_small_loader = DataLoader(test_small, batch_size=64)
    test_large_loader = DataLoader(test_large, batch_size=64)

    # -------- MODEL --------
    model = RobustGNN(hidden_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # -------- TRAIN LOOP --------
    best_val_acc = 0
    best_state = None
    patience = 20
    trigger = 0

    for epoch in range(1, 151):
        train(model, train_loader, optimizer, criterion)
        val_acc = test(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            trigger = 0
        else:
            trigger += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Val Acc: {val_acc:.4f}")

        if trigger >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # -------- FINAL TEST --------
    model.load_state_dict(best_state)

    small_acc = test(model, test_small_loader)
    large_acc = test(model, test_large_loader)

    small_accs.append(small_acc)
    large_accs.append(large_acc)

    print(f"Run {run+1} Small Test Acc: {small_acc:.4f}")
    print(f"Run {run+1} Large Test Acc: {large_acc:.4f}")

# =========================
# FINAL RESULTS
# =========================
avg_small = sum(small_accs) / NUM_RUNS
avg_large = sum(large_accs) / NUM_RUNS

print("\n==================================================")
print("Dataset: Tox21 (SR-MMP)")
print("==================================================\n")

print("Strategy")
print("Custom Split + 5 Runs (Average)\n")

print(f"Average Small Graph Accuracy: {avg_small*100:.2f}%")
print(f"Average Large Graph Accuracy: {avg_large*100:.2f}%\n")

print("Average Data Sizes:")
print(f"Train: {sum(train_sizes)/NUM_RUNS:.1f}")
print(f"Validation: {sum(val_sizes)/NUM_RUNS:.1f}")
print(f"Test: {sum(test_sizes)/NUM_RUNS:.1f}")

print("\nKey Takeaway:")
if avg_large > avg_small:
    print("Model generalizes well to large graphs.")
else:
    print("Model struggles to generalize to large graphs.")