import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
import copy
import random

class RobustGNN(torch.nn.Module):
    def __init__(self, hidden_channels, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
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

def evaluate(loader, model, device):
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
            correct += int((pred == data.y.view(-1).long()).sum())
    return correct / len(loader.dataset)

def run_experiment():

    # 🔹 LOAD DATASET
    dataset = MoleculeNet(root='data/HIV', name='HIV')
    dataset = [d for d in dataset if d.y.item() != -1]

    # 🔹 FIX DATA TYPE
    for d in dataset:
        d.x = d.x.float()

    # 🔹 SPLIT BY SIZE
    small_graphs = [d for d in dataset if d.num_nodes <= 40]
    large_graphs = [d for d in dataset if d.num_nodes > 40]

    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Total Graphs: {len(dataset)}")
    print(f"Small Graphs (≤40 nodes): {len(small_graphs)}")
    print(f"Large Graphs (>40 nodes): {len(large_graphs)}")
    print("="*60)

    split = int(len(small_graphs) * 0.8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = dataset[0].num_node_features

    all_results = []

    for run in range(5):
        print(f"\n=========== RUN {run+1} ===========")

        random.shuffle(small_graphs)

        train_data = small_graphs[:split]
        val_small_data = small_graphs[split:]

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_small_loader = DataLoader(val_small_data, batch_size=32)
        val_large_loader = DataLoader(large_graphs, batch_size=32)

        model = RobustGNN(128, in_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        best_model = None
        best_val_small = 0
        trigger = 0
        patience = 15

        # 🔹 TRAIN
        for epoch in range(1, 101):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y.view(-1).long())
                loss.backward()
                optimizer.step()

            val_small_acc = evaluate(val_small_loader, model, device)

            if val_small_acc > best_val_small:
                best_val_small = val_small_acc
                best_model = copy.deepcopy(model.state_dict())
                trigger = 0
            else:
                trigger += 1

            if trigger >= patience:
                break

        model.load_state_dict(best_model)

        print("\n" + "-"*50)
        print("FINAL TEST RESULTS (THIS RUN)")
        print("-"*50)

        train_acc = evaluate(train_loader, model, device)
        val_small_acc = evaluate(val_small_loader, model, device)
        val_large_acc = evaluate(val_large_loader, model, device)

        print(f"Train Small Graphs     | Accuracy: {train_acc*100:.2f}%")
        print(f"Validation Small Graphs| Accuracy: {val_small_acc*100:.2f}%")
        print(f"Validation Large Graphs| Accuracy: {val_large_acc*100:.2f}%")

        all_results.append((train_acc, val_small_acc, val_large_acc))

    # 🔥 AVERAGE RESULTS
    avg_train = sum(r[0] for r in all_results) / 5
    avg_val_small = sum(r[1] for r in all_results) / 5
    avg_val_large = sum(r[2] for r in all_results) / 5

    print("\n" + "="*60)
    print("FINAL AVERAGE RESULTS (5 RUNS)")
    print("="*60)
    print(f"Avg Train Small Graphs     | Accuracy: {avg_train*100:.2f}%")
    print(f"Avg Validation Small Graphs| Accuracy: {avg_val_small*100:.2f}%")
    print(f"Avg Validation Large Graphs| Accuracy: {avg_val_large*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_experiment()