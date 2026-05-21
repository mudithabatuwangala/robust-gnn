import torch
from torch_geometric.loader import DataLoader
import copy

from src.datasets.dataset_loader import load_dataset
from src.datasets.splitters import split_by_num_nodes
from src.models.rgnn import Rgnn
from src.training.trainer import train
from src.evaluation.evaluate import evaluate, evaluate_metrics


def run(config):

    # 1. LOAD DATASET
    dataset = load_dataset(config["dataset_name"], config["root"])

    # 2. SPLIT
    train_val, challenge = split_by_num_nodes(dataset, config["size_threshold"])

    split_idx = int(len(train_val) * config["train_ratio"])
    train_data = train_val[:split_idx]
    val_data = train_val[split_idx:]

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"])
    challenge_loader = DataLoader(challenge, batch_size=config["batch_size"])

    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Challenge size: {len(challenge)}")

    # 3. MODEL
    model = Rgnn(
        in_channels=dataset.num_node_features,
        hidden_channels=config["hidden_channels"],
        num_classes=dataset.num_classes,
        model_type=config["model_type"],
        pooling=config["pooling"],
        activation=config["activation"],
        num_layers=config["num_layers"],
        seeds=config["seeds"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    # 4. TRAIN LOOP
    best_val_acc = 0
    best_model = None
    trigger_times = 0

    # NEW: tracking variables
    min_loss = float("inf")
    early_stop_epoch = config["epochs"]  # default if no early stop

    print("Starting training...")

    for epoch in range(1, config["epochs"] + 1):
        loss = train(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, val_loader)

        # Track minimum loss
        if loss < min_loss:
            min_loss = loss

        # Validation tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            trigger_times = 0
        else:
            trigger_times += 1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping
        if trigger_times >= config["patience"]:
            early_stop_epoch = epoch  # record stopping point
            print(f"Early stopping at epoch {epoch}")
            break

    # If no early stopping happened
    if trigger_times < config["patience"]:
        early_stop_epoch = epoch

    # 5. FINAL TEST
    model.load_state_dict(best_model)
    challenge_acc = evaluate(model, challenge_loader)
    precision, recall, f1 = evaluate_metrics(model, challenge_loader)

    print("-" * 30)
    print(f'Final Best Val Acc: {best_val_acc:.4f}')
    print(f'Challenge Acc (Large Graphs): {challenge_acc:.4f}')
    print(f'Minimum Training Loss: {min_loss:.4f}')
    print(f'Early Stopping Epoch: {early_stop_epoch}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return {
        "best_val_acc": best_val_acc,
        "challenge_acc": challenge_acc,
        "min_loss": min_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

#     return {
#     "experiment_type": config["experiment_type"],

#     # shared metrics
#     "best_val_acc": best_val_acc,
#     "challenge_acc": challenge_acc,
#     "min_loss": min_loss,

#     # identifiers (IMPORTANT FOR CSV SPLIT)
#     "dataset": config["dataset_name"],
#     "model": config["model_type"],
#     "activation": config["activation"],
#     "pooling": config["pooling"],

#     "hidden_channels": config["hidden_channels"],
#     "num_layers": config["num_layers"],
#     "seeds": config["seeds"],
# }
