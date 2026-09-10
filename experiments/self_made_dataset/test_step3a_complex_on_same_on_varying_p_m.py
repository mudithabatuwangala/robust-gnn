import torch
import networkx as nx
import os
import numpy as np
import random
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
import matplotlib.pyplot as plt

# OMP Fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- CONFIGURATION ---
GLOBAL_SEED = 64  # Change this number (e.g., 7, 123, 999) to test different "universes"
NUM_TRIALS = 5    # How many times to repeat each test
SAMPLES = 300     # Increased from 200 for better reliability
# ---------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RobustGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(RobustGNN, self).__init__()
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = global_max_pool(x, batch)
        return self.lin(x)

def generate_complex_sample(num_nodes, graph_type, param):
    # This uses the current state of np.random (affected by set_seed)
    if graph_type == 'ER':
        G = nx.erdos_renyi_graph(num_nodes, param)
    else:
        m_val = max(1, int(param))
        G = nx.barabasi_albert_graph(num_nodes, m_val)

    is_positive = np.random.rand() > 0.5
    x = torch.zeros((num_nodes, 3)); x[:, 0] = 1 
    nodes = list(G.nodes())
    u, v, w = nodes[0], nodes[1], nodes[2]
    x[u] = torch.tensor([0, 1, 0]); x[v] = torch.tensor([0, 0, 1]) 

    if is_positive:
        G.add_edge(u, w); G.add_edge(w, v)
        if G.has_edge(u, v): G.remove_edge(u, v)
        y = 1
    else:
        if G.has_edge(u, v): G.remove_edge(u, v)
        common = list(nx.common_neighbors(G, u, v))
        for c in common: G.remove_edge(u, c)
        y = 0

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    if edge_index.numel() > 0:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([y]))

def run_test():
    set_seed(GLOBAL_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "step3_trained_model.pt"
    
    model = RobustGNN(128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    p_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    m_values = [1, 2, 3, 5, 10]
    
    print(f"--- RUNNING WITH SEED: {GLOBAL_SEED} ---")
    
    for label, sweep_type, vals in [("ER Density", "ER", p_values), ("BA Connectivity", "BA", m_values)]:
        print(f"\nSweep: {label}")
        for val in vals:
            trial_accs = []
            for _ in range(NUM_TRIALS):
                samples = [generate_complex_sample(100, sweep_type, val) for _ in range(SAMPLES)]
                loader = DataLoader(samples, batch_size=32)
                correct = 0
                for data in loader:
                    data = data.to(device)
                    with torch.no_grad():
                        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
                        correct += int((pred == data.y).sum())
                trial_accs.append((correct / SAMPLES) * 100)
            
            print(f"{sweep_type} {val:.2f} | Mean: {np.mean(trial_accs):.2f}% | Std: {np.std(trial_accs):.2f}%")

if __name__ == "__main__":
    run_test()