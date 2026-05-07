from torch_geometric.datasets import TUDataset

def load_dataset(name, root):
    return TUDataset(root=root, name=name).shuffle()