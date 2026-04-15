import torch
from torch_geometric.data import Data
import numpy as np

def audit(file_path=r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step1.pt"):
    try:
        # PyTorch 2.6+ Fix: weights_only=False
        dataset = torch.load(file_path, weights_only=False)
        
        # Reports initialization
        combo_counts = {}
        node_counts_small = []
        node_counts_large = []

        for d in dataset:
            # 1. Combination Counts (Type + Size + Label)
            g_type = d.graph_type # e.g., "ER_Small"
            label = "Pos" if d.y == 1 else "Neg"
            key = f"{g_type}_{label}"
            combo_counts[key] = combo_counts.get(key, 0) + 1

            # 2. Node Range Checking
            num_nodes = d.x.shape[0]
            if d.is_large:
                node_counts_large.append(num_nodes)
            else:
                node_counts_small.append(num_nodes)

        print("\n" + "="*50)
        print("         DETAILED UNIVERSAL DATASET AUDIT")
        print("="*50)
        print(f"Total Graphs Processed: {len(dataset)}")
        print("-" * 50)
        
        # Combination Report
        print("COMBINATION BREAKDOWN:")
        for combo in sorted(combo_counts.keys()):
            print(f"  {combo.ljust(25)} : {combo_counts[combo]} graphs")
            
        print("-" * 50)
        
        # Node Range Report
        if node_counts_small:
            print(f"SMALL GRAPHS NODE RANGE (Target: 15-25):")
            print(f"  - Min Nodes: {min(node_counts_small)}")
            print(f"  - Max Nodes: {max(node_counts_small)}")
            print(f"  - Avg Nodes: {np.mean(node_counts_small):.2f}")
        
        print("-" * 50)
        
        if node_counts_large:
            print(f"LARGE GRAPHS NODE RANGE (Target: 80-120):")
            print(f"  - Min Nodes: {min(node_counts_large)}")
            print(f"  - Max Nodes: {max(node_counts_large)}")
            print(f"  - Avg Nodes: {np.mean(node_counts_large):.2f}")
            
        print("="*50)

        # Integrity Check
        if not node_counts_small or not node_counts_large:
            print("WARNING: One of the size categories is EMPTY!")
        elif max(node_counts_small) >= min(node_counts_large):
            print("WARNING: Overlap detected between Small and Large ranges!")
        else:
            print("STATUS: Node ranges are distinct and correct.")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    audit()