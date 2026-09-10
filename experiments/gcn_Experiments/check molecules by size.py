import torch
from torch_geometric.datasets import TUDataset
import numpy as np

# Load the dataset
dataset = TUDataset(root='/tmp/Mutagenicity', name='Mutagenicity')
node_counts = [data.num_nodes for data in dataset]

# Define your categories based on the professor's interest
below_20 = sum(1 for count in node_counts if count < 20)
between_20_40 = sum(1 for count in node_counts if 20 <= count <= 40)
above_40 = sum(1 for count in node_counts if count > 40)

print("--- Molecule Size Distribution ---")
print(f"Small (Below 20 nodes):  {below_20} molecules")
print(f"Medium (20-40 nodes):    {between_20_40} molecules")
print(f"Large (Above 40 nodes):   {above_40} molecules")

# Optional: More granular bins if you want to be even more detailed
bins = [0, 10, 20, 30, 40, 50, 100, max(node_counts)]
counts, _ = np.histogram(node_counts, bins=bins)

print("\n--- Granular Breakdown ---")
for i in range(len(bins)-1):
    print(f"Nodes {bins[i]}-{bins[i+1]}: {counts[i]} molecules")