import torch
import networkx as nx
import os
import numpy as np
import random
import warnings
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

GLOBAL_SEED = 64  
NUM_TRIALS = 5    
SAMPLES_PER_SIZE = 300 

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

def generate_complex_sample(num_nodes, graph_type, param, force_shuffle=True):
    is_positive = random.choice([True, False])
    if graph_type == "ER":
        G = nx.erdos_renyi_graph(num_nodes, param)
    else:
        G = nx.barabasi_albert_graph(num_nodes, int(param))

    x = torch.zeros((num_nodes, 3))
    x[:, 0] = 1 
    
    nodes = list(G.nodes())
    if force_shuffle:
        random.shuffle(nodes)
        
    u_red, v_green, w_bridge = nodes[0], nodes[1], nodes[2]
    x[u_red] = torch.tensor([0, 1, 0])   
    x[v_green] = torch.tensor([0, 0, 1]) 

    if is_positive:
        G.add_edge(u_red, w_bridge)
        G.add_edge(w_bridge, v_green)
        if G.has_edge(u_red, v_green): G.remove_edge(u_red, v_green)
        y = 1
    else:
        if G.has_edge(u_red, v_green): G.remove_edge(u_red, v_green)
        common = list(nx.common_neighbors(G, u_red, v_green))
        for c in common:
            G.remove_edge(u_red, c)
        y = 0

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    if edge_index.numel() > 0:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    return Data(x=x, edge_index=edge_index, y=torch.tensor([y]))


# ✅ NEW: Evaluation with PRF
def evaluate(loader, model, device):
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100

    return accuracy, precision, recall, f1


def run_hard_shuffled_sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "step3_trained_model.pt"
    
    print("="*80)
    print("                HARD SHUFFLED PERMUTATION SWEEP MANIFEST             ")
    print("="*80)
    print(f"[*] TRAINED MODEL USED  : {model_path}")
    print(f"[*] TRAINED ON          : Small graphs (20 nodes)")
    print(f"[*] VALIDATED/TESTED ON : ER/BA graphs (20 & 100 nodes, shuffled)")
    print(f"[*] MODEL POOLING LAYER : Global MAX Pooling")
    print(f"[*] PERMUTATION SHUFFLE : True (Permutation invariance test)")
    print("="*80 + "\n")

    if not os.path.exists(model_path):
        print(f"Trained target weights missing at {model_path}")
        return

    model = RobustGNN(128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    p_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    m_values = [1, 2, 3, 5, 10]

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    for label, sweep_type, vals in [("ER Density", "ER", p_values), ("BA Connectivity", "BA", m_values)]:
        print(f"--- Sweeping Domain Matrix (Shuffled): {label} ---")

        for val in vals:
            print(f"\n  [Sweep Parameter: {val}]")

            for size_label, node_count in [("Small (20 nodes)", 20), ("Large (100 nodes)", 100)]:

                accs, precs, recs, f1s = [], [], [], []

                for _ in range(NUM_TRIALS):
                    samples = [
                        generate_complex_sample(node_count, sweep_type, val, force_shuffle=True)
                        for _ in range(SAMPLES_PER_SIZE)
                    ]

                    loader = DataLoader(samples, batch_size=32)

                    acc, p, r, f1 = evaluate(loader, model, device)

                    accs.append(acc)
                    precs.append(p)
                    recs.append(r)
                    f1s.append(f1)

                print(f"    -> {size_label.ljust(18)} | "
                      f"Acc: {np.mean(accs):.2f}% | "
                      f"P: {np.mean(precs):.3f} | "
                      f"R: {np.mean(recs):.3f} | "
                      f"F1: {np.mean(f1s):.3f}")

        print("\n" + "-"*80)


if __name__ == "__main__":
    run_hard_shuffled_sweep()