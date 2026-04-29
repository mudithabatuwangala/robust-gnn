import torch
import random
import numpy as np
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, degree

def get_hiv_pools():
    dataset = MoleculeNet(root='data/HIV', name='HIV')
    # Filter small building blocks (<= 40 nodes)
    small_pool = [d for d in dataset if d.num_nodes <= 40]
    
    pos_pool = [d for d in small_pool if d.y.item() == 1]
    neg_pool = [d for d in small_pool if d.y.item() == 0]
    
    print(f"Building Blocks: {len(pos_pool)} Positive, {len(neg_pool)} Negative")
    return pos_pool, neg_pool

class HIVScaleGenerator:
    def __init__(self, pos_pool, neg_pool):
        self.pos_pool = pos_pool
        self.neg_pool = neg_pool

    def generate(self, target_label, num_samples=250):
        synthetic_graphs = []
        pool = self.pos_pool if target_label == 1 else self.neg_pool
        
        for _ in range(num_samples):
            # Step 2: Combine k small graphs (k = 2 to 10)
            k = random.randint(2, 10)
            selected = [random.choice(pool) for _ in range(k)]
            
            # Step 3: Preserve internal structure via Batching
            batch = Batch.from_data_list(selected)
            x = batch.x
            edge_index = batch.edge_index
            ptr = batch.ptr
            num_nodes = x.size(0)

            # Step 4: Connectivity (Local Bridges + Long-range Edges)
            new_edges = []
            
            # Local Bridges: Ensure CC=1 by linking component i to i+1
            for i in range(len(selected) - 1):
                u = random.randint(ptr[i], ptr[i+1] - 1)
                v = random.randint(ptr[i+1], ptr[i+2] - 1)
                new_edges.append([u, v])

            # Long-range / Sparse Edges to match Target Density (~0.017)
            # HIV Large Density target: ~1.7% of total possible edges
            max_possible = num_nodes * (num_nodes - 1) / 2
            target_total_edges = int(0.017 * max_possible)
            current_edges = edge_index.size(1) / 2
            
            edges_to_add = max(0, target_total_edges - current_edges - len(new_edges))
            
            for _ in range(int(edges_to_add)):
                u = random.randint(0, num_nodes - 1)
                v = random.randint(0, num_nodes - 1)
                if u != v:
                    new_edges.append([u, v])

            if new_edges:
                bridge_t = torch.tensor(new_edges).t().contiguous()
                edge_index = torch.cat([edge_index, to_undirected(bridge_t)], dim=1)

            synthetic_graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([target_label])))
            
        return synthetic_graphs

# --- Execution ---
pos_pool, neg_pool = get_hiv_pools()
gen = HIVScaleGenerator(pos_pool, neg_pool)

# Create balanced synthetic dataset
print("Generating 250 Positive and 250 Negative Large Synthetic Graphs...")
pos_synthetic = gen.generate(target_label=1, num_samples=250)
neg_synthetic = gen.generate(target_label=0, num_samples=250)

full_synthetic_set = pos_synthetic + neg_synthetic
random.shuffle(full_synthetic_set)

# Save
torch.save(full_synthetic_set, 'hiv_synthetic_scale_data.pt')
print("\nDataset saved as 'hiv_synthetic_scale_data.pt'")

# --- Summary Statistics ---
all_nodes = [d.num_nodes for d in full_synthetic_set]
all_degrees = [degree(d.edge_index[0], d.num_nodes).mean().item() for d in full_synthetic_set]
all_densities = [(d.edge_index.size(1)/2) / (d.num_nodes * (d.num_nodes-1)/2) for d in full_synthetic_set]

print("-" * 30)
print("GENERATED DATASET SUMMARY")
print("-" * 30)
print(f"Total Graphs:      {len(full_synthetic_set)}")
print(f"Label Balance:     250 Pos / 250 Neg")
print(f"Avg Node Count:    {np.mean(all_nodes):.2f} (Target was Large)")
print(f"Avg Degree:        {np.mean(all_degrees):.2f} (HIV Target ~4.2)")
print(f"Avg Density:       {np.mean(all_densities):.4f} (HIV Target ~0.017)")
print(f"Min/Max Nodes:     {min(all_nodes)} / {max(all_nodes)}")
print("-" * 30)