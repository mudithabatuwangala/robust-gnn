import torch
from torch_geometric.data import Data
import numpy as np

def audit_complex(file_path=r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step2_complex.pt"):
    try:
        # PyTorch 2.6+ fix: weights_only=False
        dataset = torch.load(file_path, weights_only=False)
        
        stats = {}
        node_counts_small = []
        node_counts_large = []

        for d in dataset:
            g_type = d.graph_type
            label = "Pos" if d.y == 1 else "Neg"
            key = f"{g_type}_{label}"
            stats[key] = stats.get(key, 0) + 1

            num_nodes = d.x.shape[0]
            if d.is_large:
                node_counts_large.append(num_nodes)
            else:
                node_counts_small.append(num_nodes)

        print("\n" + "="*50)
        print("      COMPLEX MOLECULAR DATASET AUDIT (R-B-G)")
        print("="*50)
        print(f"Total Graphs: {len(dataset)}")
        print("-" * 50)
        for k in sorted(stats.keys()):
            print(f"{k.ljust(25)} : {stats[k]} graphs")
        
        print("-" * 50)
        print(f"Node Range Small (Target 15-25) : {min(node_counts_small)} - {max(node_counts_small)}")
        print(f"Node Range Large (Target 80-120): {min(node_counts_large)} - {max(node_counts_large)}")
        print("="*50)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    audit_complex()