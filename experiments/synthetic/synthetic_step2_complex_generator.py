import networkx as nx
import torch
import random
import os
from torch_geometric.data import Data

def create_complex_molecule_dataset(
    num_samples=2000,
    save_path=r"C:\Users\minha\OneDrive\Desktop\GNN_semester_2\synthetic dataset\synthetic_step2_complex.pt"
):
    dataset = []

    stats = {
        "Small": 0,
        "Large": 0,
        "ER": 0,
        "BA": 0,
        "Positive": 0,
        "Negative": 0
    }

    print("=" * 70)
    print(f"Generating {num_samples} complex molecular-type graphs...")
    print("=" * 70)

    for i in range(num_samples):

        size_choice = random.choice(['Small', 'Large'])
        type_choice = random.choice(['ER', 'BA'])
        is_positive = random.choice([True, False])

        num_nodes = random.randint(15, 25) if size_choice == 'Small' else random.randint(80, 120)

        # Base Graph
        if type_choice == 'ER':
            G = nx.erdos_renyi_graph(num_nodes, p=0.15)
        else:
            G = nx.barabasi_albert_graph(num_nodes, m=2)

        x = torch.zeros((num_nodes, 3))
        x[:, 0] = 1  # default class

        nodes = list(G.nodes())
        random.shuffle(nodes)

        u_red, v_green, w_bridge = nodes[0], nodes[1], nodes[2]

        x[u_red] = torch.tensor([0, 1, 0])   # Red
        x[v_green] = torch.tensor([0, 0, 1]) # Green

        if is_positive:
            G.add_edge(u_red, w_bridge)
            G.add_edge(w_bridge, v_green)
            if G.has_edge(u_red, v_green):
                G.remove_edge(u_red, v_green)
            y = 1
            stats["Positive"] += 1
        else:
            if G.has_edge(u_red, v_green):
                G.remove_edge(u_red, v_green)

            common = list(nx.common_neighbors(G, u_red, v_green))
            for c in common:
                G.remove_edge(u_red, c)

            y = 0
            stats["Negative"] += 1

        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        if edge_index.numel() > 0:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y]))
        data.is_large = (size_choice == 'Large')
        data.graph_type = f"{type_choice}_{size_choice}"

        dataset.append(data)

        stats[size_choice] += 1
        stats[type_choice] += 1

    torch.save(dataset, save_path)

    # Dataset report
    print("\n" + "=" * 70)
    print("Dataset Statistics Report")
    print("=" * 70)

    print(f"Total Samples        : {num_samples}")
    print(f"Small Graphs         : {stats['Small']}")
    print(f"Large Graphs         : {stats['Large']}")
    print(f"ER Graphs            : {stats['ER']}")
    print(f"BA Graphs            : {stats['BA']}")
    print(f"Positive Samples     : {stats['Positive']}")
    print(f"Negative Samples     : {stats['Negative']}")

    print("\nClass Balance:")
    pos_ratio = stats["Positive"] / num_samples * 100
    neg_ratio = stats["Negative"] / num_samples * 100

    print(f"Positive %           : {pos_ratio:.2f}%")
    print(f"Negative %           : {neg_ratio:.2f}%")

    print("=" * 70)
    print(f"Dataset saved to: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    create_complex_molecule_dataset()