import torch
import torch.nn.functional as F
import copy
import random
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool
)

# Config
TASK_IDX = 10
NUM_RUNS = 5

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
dataset = MoleculeNet(
    root='/tmp/Tox21',
    name='Tox21'
)

# Filter data
filtered_dataset = []

for data in dataset:
    y = data.y[:, TASK_IDX]
    # Remove missing labels
    if not torch.isnan(y):
        # Binary label
        data.y = y.long()
        # Convert node features to float
        data.x = data.x.float()
        filtered_dataset.append(data)

print(f"Filtered dataset size: {len(filtered_dataset)}")

def apply_activation(x, activation):
    if activation == "relu":
        return F.relu(x)
    elif activation == "elu":
        return F.elu(x)
    elif activation == "leaky_relu":
        return F.leaky_relu(x)

class RobustGNN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        pooling,
        activation
    ):
        super().__init__()
        self.pooling = pooling
        self.activation = activation
        self.conv1 = GCNConv(
            dataset.num_node_features,
            hidden_channels
        )
        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )
        self.conv3 = GCNConv(
            hidden_channels,
            hidden_channels
        )
        self.conv4 = GCNConv(
            hidden_channels,
            hidden_channels
        )
        self.conv5 = GCNConv(
            hidden_channels,
            hidden_channels
        )
        self.lin = torch.nn.Linear(
            hidden_channels,
            1
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        x = apply_activation(
            self.conv1(x, edge_index),
            self.activation
        )

        x = apply_activation(
            self.conv2(x, edge_index),
            self.activation
        )

        x = apply_activation(
            self.conv3(x, edge_index),
            self.activation
        )

        x = apply_activation(
            self.conv4(x, edge_index),
            self.activation
        )

        x = apply_activation(
            self.conv5(x, edge_index),
            self.activation
        )

        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)
        return self.lin(x).view(-1)

def train(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(
            data.x,
            data.edge_index,
            data.batch
        )
        loss = criterion(
            out,
            data.y.float()
        )
        loss.backward()
        optimizer.step()
        total_loss += (
            loss.item() * data.num_graphs
        )

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):

    model.eval()
    correct = 0
    total = 0
    TP = 0
    FP = 0
    FN = 0

    for data in loader:
        data = data.to(device)

        out = model(
            data.x,
            data.edge_index,
            data.batch
        )
        pred = (
            torch.sigmoid(out) > 0.5
        ).long()

        labels = data.y.long()

        correct += int(
            (pred == labels).sum()
        )

        total += labels.size(0)

        TP += int(
            ((pred == 1) & (labels == 1)).sum()
        )
        FP += int(
            ((pred == 1) & (labels == 0)).sum()
        )
        FN += int(
            ((pred == 0) & (labels == 1)).sum()
        )

    acc = correct / total if total > 0 else 0

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0 else 0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0 else 0
    )

    if (precision + recall) > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:
        f1 = 0

    return acc, precision, recall, f1

# Ssettings
poolings = [
    "mean",
    "max",
    "sum"
]

activations = [
    "relu",
    "elu",
    "leaky_relu"
]

results = []

for pooling in poolings:

    for activation in activations:
        print("\n===================================")
        print(f"Pooling: {pooling}")
        print(f"Activation: {activation}")
        print("===================================")

        run_val_accs = []
        run_large_accs = []
        run_losses = []

        run_precisions = []
        run_recalls = []
        run_f1s = []

        for run in range(NUM_RUNS):

            print(f"\nRun {run+1}")

            small_graphs = [
                d for d in filtered_dataset
                if d.num_nodes <= 40
            ]

            large_graphs = [
                d for d in filtered_dataset
                if d.num_nodes > 40
            ]

            random.shuffle(small_graphs)
            random.shuffle(large_graphs)

            # Small graph:0.6 train, 0.2 val, 0.2 test
            n_small = len(small_graphs)

            train_small = small_graphs[
                :int(0.6 * n_small)
            ]

            val_small = small_graphs[
                int(0.6 * n_small):
                int(0.8 * n_small)
            ]

            test_small = small_graphs[
                int(0.8 * n_small):
            ]

            # Large split: 
            # 20% VALIDATION
            # 80% TEST
            n_large = len(large_graphs)

            val_large = large_graphs[
                :int(0.2 * n_large)
            ]

            test_large = large_graphs[
                int(0.2 * n_large):
            ]

            # Validation contains:
            # small validation + large validation
            val_data = val_small + val_large

            train_loader = DataLoader(
                train_small,
                batch_size=64,
                shuffle=True
            )

            val_loader = DataLoader(
                val_data,
                batch_size=64
            )

            test_large_loader = DataLoader(
                test_large,
                batch_size=64
            )

            print("\n--- DATA SPLITS ---")

            print(
                f"Train Small: {len(train_small)}"
            )

            print(
                f"Validation Small: {len(val_small)}"
            )

            print(
                f"Validation Large: {len(val_large)}"
            )

            print(
                f"Test Small: {len(test_small)}"
            )

            print(
                f"Test Large: {len(test_large)}"
            )

            model = RobustGNN(
                hidden_channels=128,
                pooling=pooling,
                activation=activation
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.001
            )

            criterion = torch.nn.BCEWithLogitsLoss()

            # Train
            best_val_acc = 0
            best_state = None
            best_loss = 999999
            patience = 20
            trigger = 0

            for epoch in range(1, 151):

                loss = train(
                    model,
                    train_loader,
                    optimizer,
                    criterion
                )

                val_acc, _, _, _ = evaluate(
                    model,
                    val_loader
                )

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(
                        model.state_dict()
                    )
                    trigger = 0
                else:
                    trigger += 1

                if loss < best_loss:
                    best_loss = loss

                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch:03d}, "
                        f"Loss: {loss:.4f}, "
                        f"Val Acc: {val_acc:.4f}"
                    )

                if trigger >= patience:
                    print(
                        f"Early stopping at epoch {epoch}"
                    )
                    break
            
            #Test
            model.load_state_dict(best_state)
            large_acc, precision, recall, f1 = evaluate(
                model,
                test_large_loader
            )

            # Stire results
            run_val_accs.append(best_val_acc)
            run_large_accs.append(large_acc)
            run_losses.append(best_loss)
            run_precisions.append(precision)
            run_recalls.append(recall)
            run_f1s.append(f1)

            print("\n--- RUN RESULTS ---")
            print(
                f"Best Val Acc: {best_val_acc:.4f}"
            )
            print(
                f"Challenge Acc: {large_acc:.4f}"
            )
            print(
                f"Min Loss: {best_loss:.4f}"
            )
            print(
                f"Precision: {precision:.4f}"
            )
            print(
                f"Recall: {recall:.4f}"
            )
            print(
                f"F1 Score: {f1:.4f}"
            )

        # Average over results
        results.append({
            "Model": "gcn",
            "Pooling": pooling,
            "Activation": activation,
            "Best Val Acc":
                round(
                    sum(run_val_accs) / NUM_RUNS,
                    4
                ),
            "Challenge Acc":
                round(
                    sum(run_large_accs) / NUM_RUNS,
                    4
                ),
            "Min Loss":
                round(
                    sum(run_losses) / NUM_RUNS,
                    4
                ),
            "Precision":
                round(
                    sum(run_precisions) / NUM_RUNS,
                    4
                ),
            "Recall":
                round(
                    sum(run_recalls) / NUM_RUNS,
                    4
                ),
            "F1 Score":
                round(
                    sum(run_f1s) / NUM_RUNS,
                    4
                )
        })


print("Final Table")

header = (
    f"{'Model':<10}"
    f"{'Pooling':<10}"
    f"{'Activation':<15}"
    f"{'Best Val Acc':<15}"
    f"{'Challenge Acc':<18}"
    f"{'Min Loss':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<10}"
    f"{'F1 Score':<10}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['Model']:<10}"
        f"{r['Pooling']:<10}"
        f"{r['Activation']:<15}"
        f"{r['Best Val Acc']:<15}"
        f"{r['Challenge Acc']:<18}"
        f"{r['Min Loss']:<12}"
        f"{r['Precision']:<12}"
        f"{r['Recall']:<10}"
        f"{r['F1 Score']:<10}"
    )

# Save on txt file
with open(
    "tox21_results_table.txt",
    "w"
) as f:
    f.write(header + "\n")
    f.write(
        "-" * len(header) + "\n"
    )
    for r in results:
        line = (
            f"{r['Model']:<10}"
            f"{r['Pooling']:<10}"
            f"{r['Activation']:<15}"
            f"{r['Best Val Acc']:<15}"
            f"{r['Challenge Acc']:<18}"
            f"{r['Min Loss']:<12}"
            f"{r['Precision']:<12}"
            f"{r['Recall']:<10}"
            f"{r['F1 Score']:<10}"
        )
        f.write(line + "\n")

print("\nResults saved as tox21_results_table.txt")