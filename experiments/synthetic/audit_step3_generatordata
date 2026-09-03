import torch
from torch_geometric.data import DataLoader
import networkx as nx
from torch_geometric.utils import to_networkx

def audit_dataset(file_path):
    print(f"--- Dataset Audit: {file_path} ---")
    
    if not torch.os.path.exists(file_path):
        print("Error: File nahi mili!")
        return

    dataset = torch.load(file_path, weights_only=False)
    total = len(dataset)
    
    positives = sum(1 for d in dataset if d.y.item() == 1)
    negatives = total - positives
    
    # Graphs ki details check karne ke liye
    avg_nodes = sum(d.num_nodes for d in dataset) / total
    avg_edges = sum(d.num_edges for d in dataset) / (total * 2) # Undirected isliye /2
    
    print(f"Total Samples: {total}")
    print(f"Positive Samples (y=1): {positives} ({positives/total*100:.1f}%)")
    print(f"Negative Samples (y=0): {negatives} ({negatives/total*100:.1f}%)")
    print(f"Average Nodes: {avg_nodes:.2f}")
    print(f"Average Edges per Graph: {avg_edges:.2f}")

    # Density Check
    densities = []
    for d in dataset:
        n = d.num_nodes
        e = d.num_edges / 2
        max_edges = n * (n - 1) / 2
        densities.append(e / max_edges)
    
    print(f"Min Density: {min(densities):.4f}")
    print(f"Max Density: {max(densities):.4f}")
    print(f"Avg Density: {sum(densities)/len(densities):.4f}")

    # Logic Check on a few samples
    print("\n--- Structural Sample Check ---")
    for i in range(5):
        d = dataset[i]
        # Red node index (feature [0,1,0])
        red_node = (d.x[:, 1] == 1).nonzero(as_tuple=True)[0].item()
        # Green node index (feature [0,0,1])
        green_node = (d.x[:, 2] == 1).nonzero(as_tuple=True)[0].item()
        
        # Calculate shortest path using NetworkX
        G = to_networkx(d, to_undirected=True)
        try:
            path_len = nx.shortest_path_length(G, source=red_node, target=green_node)
        except nx.NetworkXNoPath:
            path_len = "No Path"
            
        print(f"Sample {i} | Label: {d.y.item()} | Shortest Path R-G: {path_len}")

if __name__ == "__main__":
    # Apni file ka sahi path yahan likhein
    path = "step3_6c_data_small_complex.pt"
    audit_dataset(path)
