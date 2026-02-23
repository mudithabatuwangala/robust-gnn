import torch
from torch_geometric.data import Data

# 1. Define the connections (edges)
# Node 0 -> 1, 1 -> 0, 1 -> 2, 2 -> 1
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

# 2. Define node features (3 nodes, each with 1 feature)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# 3. Create the Graph Data object
data = Data(x=x, edge_index=edge_index)

print("--- PyTorch Geometric Success! ---")
print(f"Graph summary: {data}")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")