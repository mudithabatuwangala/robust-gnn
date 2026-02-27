import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv

# 1. Define the connections (edges)
# Node 0 -> 1, 1 -> 0, 1 -> 2, 2 -> 1 (nodes: 0, 1, 2)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

# 2. Define node features (3 nodes, each with 1 feature)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# 3. Create the Graph Data object
data = Data(x=x, edge_index=edge_index)

# Convert the PyG data object into NetworkX
G = to_networkx(data, to_undirected=True) 

# Initialize the layer
# Imput features of graph = 1 (x has 1 col)
# Output features = 2 (srbitrary)
conv = GCNConv(in_channels=1, out_channels=2)

# Pass the dat through the GCN layer
# We need both features x and the structure (edge_index)
output = conv(data.x, data.edge_index)

print("--- PyTorch Geometric Success! ---")
print(f"Graph summary: {data}")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")

print("Output features after one GCN layer:")
print(output)

# Draw the graph G
plt.figure(figsize=(4, 4))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=800, font_weight='bold')
plt.show()