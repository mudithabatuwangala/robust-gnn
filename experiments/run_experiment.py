import torch
from torch_geometric.loader import DataLoader
import copy

from src.datasets.dataset_loader import load_dataset
from src.datasets.splitters import split_by_num_nodes
from src.models.rgnn import Rgnn
from src.training.trainer import train
from src.evaluation.evaluate import evaluate


def run(config):

    dataset = load_dataset(config["dataset_name"], config["root"])
    # Split
    train_small_test_, large_test_data = split_by_num_nodes(dataset, config["size_threshold"])
    if config["dataset_name"] == "hiv":
        train_small_test_ = train_small_test_[:10000]
        large_test_data = large_test_data[:6000]

    tst_split_idx = int(len(train_small_test_) * config["train_ratio"])
    train_val_ = train_small_test_[:tst_split_idx]
    small_test_data = train_small_test_[tst_split_idx:]

    val_split_idx = int(len(train_val_) * config["train_ratio"])
    train_data = train_val_[:val_split_idx]
    val_data = train_val_[val_split_idx:]

    # if len(small_test_data) > len(large_test_data):
    #     small_test_data = small_test_data[:len(large_test_data)]
    # else:
    #     large_test_data = large_test_data[:len(small_test_data)]

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"])
    small_test_loader = DataLoader(small_test_data, batch_size=config["batch_size"])
    large_test_loader = DataLoader(large_test_data, batch_size=config["batch_size"])

    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Small Test size: {len(small_test_data)}, Large Test size: {len(large_test_data)}")

    model = Rgnn(
        in_channels=dataset[0].num_node_features if config["dataset_name"] == "hiv" else dataset.num_node_features,
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

    # Train loop
    best_val_acc = 0
    best_model = None
    trigger_times = 0

    # Tracking variables
    min_loss = float("inf")
    early_stop_epoch = config["epochs"]  # default if no early stop

    print("Starting training...")

    for epoch in range(1, config["epochs"] + 1):
        loss = train(model, train_loader, optimizer, criterion)
        val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader)

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

    # Test
    model.load_state_dict(best_model)
    large_acc, large_precision, large_recall, large_f1 = evaluate(model, large_test_loader)
    small_acc, small_precision, small_recall, small_f1 = evaluate(model, small_test_loader)

    print("-" * 30)
    print(f'Final Best Val Acc: {best_val_acc:.4f}')
    print(f'Small Acc: {small_acc:.4f}')
    print(f'Large Acc: {large_acc:.4f}')
    print(f'Minimum Training Loss: {min_loss:.4f}')
    print(f'Early Stopping Epoch: {early_stop_epoch}')
    print(f'Precision (Small): {small_precision:.4f}')
    print(f'Recall (Small): {small_recall:.4f}')
    print(f'F1 Score (Small): {small_f1:.4f}')
    print(f'Precision (Large): {large_precision:.4f}')
    print(f'Recall (Large): {large_recall:.4f}')
    print(f'F1 Score (Large): {large_f1:.4f}')

    return {
        "best_val_acc": best_val_acc,
        "small_acc": small_acc,
        "large_acc": large_acc,
        "min_loss": min_loss,
        "small_precision": small_precision,
        "small_recall": small_recall,
        "small_f1_score": small_f1,
        "large_precision": large_precision,
        "large_recall": large_recall,
        "large_f1_score": large_f1,
    }
