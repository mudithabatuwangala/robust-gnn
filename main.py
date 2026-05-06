# import torch
# import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.data import Data
# from torch_geometric.utils import to_networkx
# from torch_geometric.nn import GCNConv

# # 1. Define the connections (edges)
# # Node 0 -> 1, 1 -> 0, 1 -> 2, 2 -> 1 (nodes: 0, 1, 2)
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)

# # 2. Define node features (3 nodes, each with 1 feature)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# # 3. Create the Graph Data object
# data = Data(x=x, edge_index=edge_index)

# print("--- PyTorch Geometric Success! ---")
# print(f"Graph summary: {data}")
# print(f"Number of nodes: {data.num_nodes}")
# print(f"Number of edges: {data.num_edges}")

# # ---------------------------------------------------------

# # Initialize the layer
# # Imput features of graph = 1 (x has 1 col)
# # Output features = 2 (srbitrary)
# conv = GCNConv(in_channels=1, out_channels=2)

# # Pass the dat through the GCN layer
# # We need both features x and the structure (edge_index)
# output = conv(data.x, data.edge_index)

# print("Output features after one GCN layer:")
# print(output)

# # ---------------------------------------------------------

# # Convert the PyG data object into NetworkX
# G = to_networkx(data, to_undirected=True) 

# # Draw the graph G
# plt.figure(figsize=(4, 4))
# nx.draw(G, with_labels=True, node_color='skyblue', node_size=800, font_weight='bold')
# plt.show()


import argparse
import yaml
# import itertools
# import copy

# from src.runner.repeat_runner import run_multiple_times
# from utils.advanced_plotting import plot_boxplots
# # from experiments.run_experiment import run
# from src.utils.aggr_results import aggregate_results
# from src.runner.experiment_runner import execute


# def load_yaml(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def merge_configs(defaults, override):
#     config = defaults.copy()
#     config.update(override)
#     return config


# def generate_configs(config):
#     keys = []
#     values = []

#     for k, v in config.items():
#         if isinstance(v, list):
#             keys.append(k)
#             values.append(v)

#     if not keys:
#         return [config]

#     combinations = list(itertools.product(*values))

#     configs = []
#     for combo in combinations:
#         new_config = copy.deepcopy(config)
#         for i, key in enumerate(keys):
#             new_config[key] = combo[i]
#         configs.append(new_config)

#     return configs


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True)

#     args = parser.parse_args()

#     # Load configs
#     global_config = load_yaml("configs/defaults/global.yaml")
#     exp_config = load_yaml(args.config)

#     # Merge
#     merged_config = merge_configs(global_config, exp_config)

#     # Sweep support
#     configs = generate_configs(merged_config)

#     print(f"Running {len(configs)} experiments...\n")

#     for i, cfg in enumerate(configs):
#         print(f"\n===== Experiment {i+1}/{len(configs)} =====")
#         print(cfg)
#         # execute(cfg)
#         results = run_multiple_times(execute, cfg, num_runs=2)

#     aggregated = aggregate_results(results)

#     print("\nAggregated Results:")
#     for k, v in aggregated.items():
#         print(f"{k}: {v}")

#     plot_boxplots(aggregated, title="5-Run Experiment Boxplots")


import argparse
import yaml
import itertools
import copy

from src.runner.repeat_runner import run_multiple_times
from src.runner.experiment_runner import execute

from src.utils.result_collector import collect_results
from src.utils.summary_table import build_table
from src.utils.advanced_plotting import plot_dataset_results


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(defaults, override):
    config = defaults.copy()
    config.update(override)
    return config


def generate_configs(config):
    keys = []
    values = []

    for k, v in config.items():
        if isinstance(v, list):
            keys.append(k)
            values.append(v)

    if not keys:
        return [config]

    combinations = list(itertools.product(*values))

    configs = []
    for combo in combinations:
        new_config = copy.deepcopy(config)
        for i, key in enumerate(keys):
            new_config[key] = combo[i]
        configs.append(new_config)

    return configs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # -------------------------
    # LOAD CONFIGS
    # -------------------------
    global_config = load_yaml("configs/defaults/global.yaml")
    exp_config = load_yaml(args.config)

    merged_config = merge_configs(global_config, exp_config)

    configs = generate_configs(merged_config)

    print(f"\nRunning {len(configs)} experiment configs...\n")

    all_runs = []

    # -------------------------
    # RUN EXPERIMENTS
    # -------------------------
    for i, cfg in enumerate(configs):

        print(f"\n===== Experiment {i+1}/{len(configs)} =====")
        print(cfg)

        run_results = run_multiple_times(execute, cfg, num_runs=1)

        # store all runs with config
        for r in run_results:
            all_runs.append((cfg, r))

    # -------------------------
    # GROUP RESULTS
    # -------------------------
    grouped = collect_results(all_runs)

    # -------------------------
    # SAVE TABLE
    # -------------------------
    df = build_table(grouped)
    df.to_csv("results/summary_table.csv", index=False)

    print("\nSaved summary table to results/summary_table.csv")

    # # -------------------------
    # # PLOT PER DATASET
    # # -------------------------
    # for dataset_name in grouped:
    #     print(f"\nGenerating plot for dataset: {dataset_name}")
    #     plot_dataset_results(dataset_name, grouped[dataset_name])