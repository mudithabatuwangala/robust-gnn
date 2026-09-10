from torch_geometric.datasets import TUDataset, MoleculeNet

def load_dataset(name, root):
    name = name.lower()
    if name == "mutagenicity":
        dataset = TUDataset(root=root, name="MUTAGENICITY").shuffle()
    elif name == "proteins":
        dataset = TUDataset(root=root, name="PROTEINS").shuffle()
    elif name == "hiv":
        dataset = MoleculeNet(root=root, name="HIV")
        # remove invalid labels
        dataset = [
            d for d in dataset
            if d.y is not None and (d.y != -1).all()
        ]
    else:
        dataset = TUDataset(root=root, name=name.upper()).shuffle()

    # Seature standarization
    for d in dataset:
        d.x = d.x.float()
        # Label standarization
        # Scalar labels (Mutagenicity, PROTEINS, HIV after cleaning)
        if d.y.dim() == 1 or d.y.numel() == 1:
            d.y = d.y.view(-1).long()
        # Multi-label (future datasets like Tox21)
        else:
            d.y = d.y.float()
    return dataset