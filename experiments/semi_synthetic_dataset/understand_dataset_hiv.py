import torch
import random
import numpy as np
import networkx as nx
from ogb.graphproppred import PygGraphPropPredDataset

# -----------------------------
# FIX PyTorch 2.6 loading issue
# -----------------------------
if not hasattr(torch, "_orig_load"):
    torch._orig_load = torch.load

torch.load = lambda *args, **kwargs: torch._orig_load(
    *args, **{**kwargs, "weights_only": False}
)

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

print("="*50)
print("TOTAL GRAPHS:", len(dataset))
print("NODE FEATURE DIM:", dataset.num_node_features)
print("NUM CLASSES:", dataset.num_classes)
print("="*50)

# -----------------------------
# SPLIT SMALL vs LARGE
# -----------------------------
small_graphs = [d for d in dataset if d.num_nodes <= 40]
large_graphs = [d for d in dataset if d.num_nodes > 100]

print(f"Small graphs: {len(small_graphs)}")
print(f"Large graphs: {len(large_graphs)}")

# -----------------------------
# BASIC STATS FUNCTION
# -----------------------------
def compute_stats(graphs):
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.num_edges for g in graphs]

    avg_nodes = np.mean(num_nodes)
    avg_edges = np.mean(num_edges)
    avg_degree = np.mean([(2*g.num_edges)/g.num_nodes for g in graphs])

    return avg_nodes, avg_edges, avg_degree

# -----------------------------
# PRINT STATS
# -----------------------------
print("\n--- SMALL GRAPH STATS ---")
s_nodes, s_edges, s_deg = compute_stats(small_graphs)
print(f"Avg nodes: {s_nodes:.2f}")
print(f"Avg edges: {s_edges:.2f}")
print(f"Avg degree: {s_deg:.2f}")

print("\n--- LARGE GRAPH STATS ---")
l_nodes, l_edges, l_deg = compute_stats(large_graphs)
print(f"Avg nodes: {l_nodes:.2f}")
print(f"Avg edges: {l_edges:.2f}")
print(f"Avg degree: {l_deg:.2f}")

# -----------------------------
# DEGREE DISTRIBUTION (sample)
# -----------------------------
def degree_distribution(graph):
    deg = torch.zeros(graph.num_nodes)
    for i in range(graph.num_nodes):
        deg[i] = (graph.edge_index[0] == i).sum()
    return deg.numpy()

print("\n--- DEGREE DISTRIBUTION (sample) ---")
sample_small = random.choice(small_graphs)
sample_large = random.choice(large_graphs)

print("Small graph degrees:", degree_distribution(sample_small)[:10])
print("Large graph degrees:", degree_distribution(sample_large)[:10])

# -----------------------------
# CONNECTED COMPONENTS
# -----------------------------
def count_components(graph):
    G = nx.Graph()
    edges = graph.edge_index.t().tolist()
    G.add_edges_from(edges)
    return nx.number_connected_components(G)

print("\n--- CONNECTED COMPONENTS ---")
print("Small graph components:", count_components(sample_small))
print("Large graph components:", count_components(sample_large))

# -----------------------------
# DENSITY
# -----------------------------
def density(graph):
    n = graph.num_nodes
    e = graph.num_edges
    return e / (n * (n - 1) + 1e-6)

print("\n--- DENSITY ---")
print("Small density:", density(sample_small))
print("Large density:", density(sample_large))

# -----------------------------
# LABEL DISTRIBUTION
# -----------------------------
labels = [int(d.y.item()) for d in dataset]

print("\n--- LABEL DISTRIBUTION ---")
print("Positive (1):", sum(labels))
print("Negative (0):", len(labels) - sum(labels))

print("\nDONE.")