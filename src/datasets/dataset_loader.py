# from torch_geometric.datasets import TUDataset, MoleculeNet

# def load_dataset(name, root):
#     # if name.lower() == "mutagenicity":
#     #     return TUDataset(root=root, name=name).shuffle()
#     match name.lower():
#         case "mutagenicity":
#             return TUDataset(root=root, name=name).shuffle()
#         case "protein":
#             return TUDataset(root=root, name=name).shuffle()
#         case "hiv":
#             dataset = MoleculeNet(root='data/HIV', name='HIV')
#             dataset = [d for d in dataset if d.y.item() != -1]
#             return dataset
#         case _:
#             return TUDataset(root=root, name=name).shuffle()


# from torch_geometric.datasets import TUDataset, MoleculeNet


# def load_dataset(name, root):

#     name = name.lower()

#     if name == "mutagenicity":
#         dataset = TUDataset(root=root, name="MUTAGENICITY").shuffle()

#     elif name == "proteins":
#         dataset = TUDataset(root=root, name="PROTEINS").shuffle()

#     elif name == "hiv":
#         dataset = MoleculeNet(root=root, name="HIV")
#         # remove invalid labels safely
#         dataset = [d for d in dataset if d.y is not None and (d.y != -1).all()]

#     else:
#         dataset = TUDataset(root=root, name=name.upper()).shuffle()

#     # ---------------------------
#     # STANDARDIZE FEATURES
#     # ---------------------------
#     for d in dataset:
#         d.x = d.x.float()

#     return dataset


from torch_geometric.datasets import TUDataset, MoleculeNet


def load_dataset(name, root):

    name = name.lower()

    # -------------------------
    # LOAD DATASET
    # -------------------------
    if name == "mutagenicity":
        dataset = TUDataset(root=root, name="MUTAGENICITY").shuffle()

    elif name == "proteins":
        dataset = TUDataset(root=root, name="PROTEINS").shuffle()

    elif name == "hiv":
        dataset = MoleculeNet(root=root, name="HIV")

        # remove invalid labels safely
        dataset = [
            d for d in dataset
            if d.y is not None and (d.y != -1).all()
        ]

    else:
        dataset = TUDataset(root=root, name=name.upper()).shuffle()

    # -------------------------
    # FEATURE STANDARDIZATION
    # -------------------------
    for d in dataset:
        d.x = d.x.float()

        # -------------------------
        # LABEL STANDARDIZATION
        # -------------------------

        # case 1: scalar labels (Mutagenicity, PROTEINS, HIV after cleaning)
        if d.y.dim() == 1 or d.y.numel() == 1:
            d.y = d.y.view(-1).long()

        # case 2: multi-label (future datasets like Tox21)
        else:
            d.y = d.y.float()

    return dataset