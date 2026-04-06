import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def prepare_size_datasets(dataset_name='Mutagenicity'):
    """Filters data into small training sets and large test sets."""
    full_dataset = TUDataset(root='./data', name=dataset_name)
    
    # Filter by node count
    small_graphs = [data for data in full_dataset if data.num_nodes < 25]
    large_graphs = [data for data in full_dataset if data.num_nodes > 40]
    
    # Split Small Graphs: 80% train, 20% test
    torch.manual_seed(123)
    indices = torch.randperm(len(small_graphs))
    split = int(len(small_graphs) * 0.8)
    
    train_data = [small_graphs[i] for i in indices[:split]]
    test_small_data = [small_graphs[i] for i in indices[split:]]
    
    return train_data, test_small_data, large_graphs, full_dataset.num_node_features, full_dataset.num_classes

def train(model, loader, optimizer, criterion):
    """One round of the AI studying the small molecules."""
    model.train()
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def evaluate(model, loader):
    """Check how many molecules the AI identifies correctly."""
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)