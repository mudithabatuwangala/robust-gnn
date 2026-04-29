import torch

# FIX PyTorch 2.6 OGB compatibility issue
old_torch_load = torch.load

def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return old_torch_load(*args, **kwargs)

torch.load = patched_load
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset

# -----------------------------
# 1. Load dataset
# -----------------------------
dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

print("Total graphs:", len(dataset))

# -----------------------------
# 2. Basic dataset info
# -----------------------------
sample = dataset[0]

print("\n--- SAMPLE GRAPH INFO ---")
print("Num nodes:", sample.num_nodes)
print("Num edges:", sample.edge_index.shape[1])
print("Node feature dim:", sample.x.shape[1])
print("Label:", sample.y.item())

# -----------------------------
# 3. Collect statistics
# -----------------------------
node_counts = []
edge_counts = []
labels = []

for g in dataset:
    node_counts.append(g.num_nodes)
    edge_counts.append(g.edge_index.shape[1])
    labels.append(int(g.y.item()))

node_counts = np.array(node_counts)
edge_counts = np.array(edge_counts)
labels = np.array(labels)

# -----------------------------
# 4. Print statistics
# -----------------------------
print("\n--- NODE STATS ---")
print("Min:", node_counts.min())
print("Max:", node_counts.max())
print("Mean:", round(node_counts.mean(), 2))

print("\n--- EDGE STATS ---")
print("Min:", edge_counts.min())
print("Max:", edge_counts.max())
print("Mean:", round(edge_counts.mean(), 2))

print("\n--- LABEL DISTRIBUTION ---")
print("Positive (1):", (labels == 1).sum())
print("Negative (0):", (labels == 0).sum())