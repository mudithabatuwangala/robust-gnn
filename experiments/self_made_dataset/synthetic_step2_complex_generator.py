import networkx as nx
import torch
import random
import os
from torch_geometric.data import Data

def create_complex_molecule_dataset(num_samples=2000, save_path=r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step2_complex.pt"):
    dataset = []
    print(f"Generating {num_samples} complex molecular-type graphs...")

    for i in range(num_samples):
        size_choice = random.choice(['Small', 'Large'])
        type_choice = random.choice(['ER', 'BA'])
        is_positive = random.choice([True, False]) # Logic: Red -> Blue -> Green

        num_nodes = random.randint(15, 25) if size_choice == 'Small' else random.randint(80, 120)
        
        # Base Graph
        if type_choice == 'ER':
            G = nx.erdos_renyi_graph(num_nodes, p=0.15) # Slightly sparser for chains
        else:
            G = nx.barabasi_albert_graph(num_nodes, m=2)

        x = torch.zeros((num_nodes, 3))
        x[:, 0] = 1 # Default: All Blue
        
        nodes = list(G.nodes())
        random.shuffle(nodes)
        u_red, v_green, w_bridge = nodes[0], nodes[1], nodes[2]
        
        x[u_red] = torch.tensor([0, 1, 0])   # Red
        x[v_green] = torch.tensor([0, 0, 1]) # Green
        # Note: w_bridge stays Blue (index 0 is 1)

        if is_positive:
            # Force the molecular chain: Red -- Blue -- Green
            G.add_edge(u_red, w_bridge)
            G.add_edge(w_bridge, v_green)
            # Ensure no direct shortcut (to make it strictly 2-hop)
            if G.has_edge(u_red, v_green): G.remove_edge(u_red, v_green)
            y = 1
        else:
            # Negative: Ensure the chain Red-Blue-Green DOES NOT exist
            # Break any direct path through a single blue node
            if G.has_edge(u_red, v_green): G.remove_edge(u_red, v_green)
            
            # Find any common neighbors and break one side of the link
            common = list(nx.common_neighbors(G, u_red, v_green))
            for c in common:
                G.remove_edge(u_red, c)
            y = 0

        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        if edge_index.numel() > 0:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y]))
        data.is_large = (size_choice == 'Large')
        data.graph_type = f"{type_choice}_{size_choice}"
        dataset.append(data)

    torch.save(dataset, save_path)
    print(f"Complex Dataset saved to: {save_path}")

if __name__ == "__main__":
    create_complex_molecule_dataset()