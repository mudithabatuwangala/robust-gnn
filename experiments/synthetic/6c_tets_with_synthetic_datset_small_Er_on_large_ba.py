import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn.functional as F
import os
import numpy as np
import copy
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


def evaluate(model, loader, device):
    preds, targets = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)

            preds.extend(pred.cpu().numpy())
            targets.extend(data.y.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    acc = (preds == targets).mean() * 100
    p, r, f1, _ = precision_recall_fscore_support(
        targets, preds, average='binary', zero_division=0
    )

    return acc, p * 100, r * 100, f1 * 100


def run_isolated_challenge_test():
    dataset_path = r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step1.pt"

    print("="*70)
    print("                   EVALUATION MANIFEST & REPORT              ")
    print("="*70)
    print(f"[*] TRAINED EXCLUSIVELY ON: Small ER Graphs")
    print(f"[*] VALIDATED ON        : Small ER Graphs (val split)")
    print(f"[*] TEST CHALLENGE TARGT: Large BA Graphs (domain shift)")
    print(f"[*] MODEL POOLING LAYER : Global ADD Pooling")
    print("="*70 + "\n")

    if not os.path.exists(dataset_path):
        print("Dataset missing.")
        return

    master_dataset = master_dataset = torch.load(dataset_path, weights_only=False)

    train_pool, challenge_pool = [], []

    for d in master_dataset:
        is_large = getattr(d, 'is_large', d.x.shape[0] > 40)
        g_type = getattr(d, 'graph_type', 'ER' if 'ER' in str(d) else 'BA')

        if not is_large and "ER" in g_type:
            train_pool.append(d)
        elif is_large and "BA" in g_type:
            challenge_pool.append(d)

    split = int(len(train_pool) * 0.8)
    train_data = train_pool[:split]
    val_data = train_pool[split:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RobustGNN(128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    best_val_acc = 0
    best_weights = None

    # Training
    for epoch in range(1, 51):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_acc, _, _, _ = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()

    print("--- VALIDATION (Small ER) ---")
    v_acc, v_p, v_r, v_f1 = evaluate(model, val_loader, device)
    print(f"Small ER | Acc: {v_acc:.2f}% | P: {v_p:.2f}% | R: {v_r:.2f}% | F1: {v_f1:.2f}%")

    print("\n--- CHALLENGE TEST (Large BA) ---")

    ch_loader = DataLoader(challenge_pool, batch_size=32)

    NUM_TRIALS = 5
    accs, ps, rs, f1s = [], [], [], []

    for _ in range(NUM_TRIALS):
        acc, p, r, f1 = evaluate(model, ch_loader, device)
        accs.append(acc)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)

    print(
        f"Large BA | "
        f"Acc: {np.mean(accs):.2f}% (±{np.std(accs):.2f}) | "
        f"P: {np.mean(ps):.2f}% | "
        f"R: {np.mean(rs):.2f}% | "
        f"F1: {np.mean(f1s):.2f}%"
    )

    print("="*70)


if __name__ == "__main__":
    run_isolated_challenge_test()