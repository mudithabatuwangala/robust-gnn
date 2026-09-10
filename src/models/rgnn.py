import torch
import torch.nn.functional as F
# import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_max_pool, global_mean_pool, global_add_pool

ACTIVATIONS = {
    "relu": F.relu,
    "elu": F.elu,
    "leaky_relu": F.leaky_relu,
}

ACTIVATION_LAYERS = {
    "relu": torch.nn.ReLU,
    "elu": torch.nn.ELU,
    "leaky_relu": torch.nn.LeakyReLU,
}

POOLING = {
    "max": global_max_pool,
    "mean": global_mean_pool,
    "sum": global_add_pool,
}

CONVS = {
    "gat": GATConv,
    "gcn": GCNConv,
    "gin": GINConv,
}

class Rgnn(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes,
                 model_type="gat",
                 pooling="max",
                 activation="relu",
                 num_layers=5, 
                 seeds=42):
        super().__init__()

        torch.manual_seed(seeds)

        Conv = CONVS[model_type]
        self.activation = ACTIVATIONS[activation]
        self.pool = POOLING[pooling]

        def mlp(in_dim, out_dim):
            activation_layer = ACTIVATION_LAYERS[activation]
            return torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim),
                activation_layer(),
                torch.nn.Linear(out_dim, out_dim)
            )

        self.convs = torch.nn.ModuleList()
        if model_type == "gin":
            self.convs.append(Conv(mlp(in_channels, hidden_channels)))
        else:
            self.convs.append(Conv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            if model_type == "gin":
                self.convs.append(Conv(mlp(hidden_channels, hidden_channels)))
            else:
                self.convs.append(Conv(hidden_channels, hidden_channels))

        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))

        x = self.pool(x, batch)
        return self.lin(x)