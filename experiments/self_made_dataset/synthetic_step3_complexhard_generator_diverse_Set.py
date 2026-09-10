import networkx as nx
import torch
import random
import numpy as np
from torch_geometric.data import Data

def create_robust_training_dataset(num_samples=2500, save_path="step3_6c_data_small_complex.pt"):
    dataset = []
    print(f"Generating {num_samples} Robust Training Samples (Small Graphs)...")

    for i in range(num_samples):
        # 1. Randomize Topology (The "Diverse" Part)
        num_nodes = random.randint(18, 30) # Always small for training
        graph_type = random.choice(['ER', 'BA'])
        
        # Wide range of density to force noise-robustness
        p_val = random.uniform(0.05, 0.45) 
        m_val = random.randint(1, 4)

        if graph_type == 'ER':
            G = nx.erdos_renyi_graph(num_nodes, p_val)
        else:
            G = nx.barabasi_albert_graph(num_nodes, m_val)

        # 2. Logic Setup (6c Base)
        x = torch.zeros((num_nodes, 3))
        x[:, 0] = 1 # Default: Blue
        
        nodes = list(G.nodes())
        random.shuffle(nodes)
        u_red, v_green = nodes[0], nodes[1]
        
        x[u_red] = torch.tensor([0, 1, 0])   # Red Node
        x[v_green] = torch.tensor([0, 0, 1]) # Green Node

        # 3. Decision: Positive (2-hop) vs Negative (Everything else)
        rand_val = random.random()
        
        if rand_val > 0.6: # 40% Positive Samples
            # Force: Red -- Blue Bridge -- Green
            w_bridge = nodes[2]
            G.add_edge(u_red, w_bridge)
            G.add_edge(w_bridge, v_green)
            # Remove direct shortcut (strictly 2-hop)
            if G.has_edge(u_red, v_green): G.remove_edge(u_red, v_green)
            y = 1
            
        elif rand_val > 0.3: # 30% "Hard Negatives" (Longer chains)
            # Red -- Blue -- Blue -- Green (3-hop)
            w1, w2 = nodes[2], nodes[3]
            G.add_edge(u_red, w1)
            G.add_edge(w1, w2)
            G.add_edge(w2, v_green)
            # Remove any 2-hop shortcuts to make it a Hard Negative
            common = list(nx.common_neighbors(G, u_red, v_green))
            for c in common: G.remove_edge(u_red, c)
            y = 0
            
        else: # 30% Simple Negatives (Disconnected)
            # Ensure no 2-hop path exists
            if G.has_edge(u_red, v_green): G.remove_edge(u_red, v_green)
            common = list(nx.common_neighbors(G, u_red, v_green))
            for c in common: G.remove_edge(u_red, c)
            y = 0

        # 4. Finalize PyG Data Object
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        if edge_index.numel() > 0:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y]))
        dataset.append(data)

    torch.save(dataset, save_path)
    print(f"Robust Dataset saved as: {save_path}")

if __name__ == "__main__":
    create_robust_training_dataset()