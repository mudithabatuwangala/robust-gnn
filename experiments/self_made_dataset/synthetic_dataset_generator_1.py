import networkx as nx
import torch
import random
import os
from torch_geometric.data import Data

def create_universal_dataset(num_samples=2000, save_path=r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step1.pt"):
    dataset = []
    print(f"Generating {num_samples} graphs with all combinations...")

    # Directory check
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for i in range(num_samples):
        # 1. Randomly choose Size (Small vs Large)
        size_choice = random.choice(['Small', 'Large'])
        # 2. Randomly choose Type (ER vs BA)
        type_choice = random.choice(['ER', 'BA'])
        # 3. Randomly choose Label (Pos vs Neg)
        is_positive = random.choice([True, False])

        # Size Logic
        if size_choice == 'Small':
            num_nodes = random.randint(15, 25)
            is_large = False
        else:
            num_nodes = random.randint(80, 120)
            is_large = True

        # Graph Type Logic
        if type_choice == 'ER':
            G = nx.erdos_renyi_graph(num_nodes, p=0.25)
        else:
            # m=2 for Barabasi
            G = nx.barabasi_albert_graph(num_nodes, m=2)

        # Features Initialize (All Blue)
        x = torch.zeros((num_nodes, 3))
        x[:, 0] = 1 # Blue index
        
        nodes = list(G.nodes())
        random.shuffle(nodes)
        u, v = nodes[0], nodes[1]
        x[u] = torch.tensor([0, 1, 0]) # Red
        x[v] = torch.tensor([0, 0, 1]) # Green

        if is_positive:
            G.add_edge(u, v) # Direct Connection
            y = 1
        else:
            if G.has_edge(u, v): G.remove_edge(u, v)
            y = 0

        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        if edge_index.numel() > 0:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Metadata storage for Audit
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y]))
        data.is_large = is_large
        data.graph_type = f"{type_choice}_{size_choice}" # e.g., "ER_Large"
        dataset.append(data)

    torch.save(dataset, save_path)
    print(f"Success! Universal dataset saved to: {save_path}")

if __name__ == "__main__":
    create_universal_dataset()