import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool, global_add_pool

ACTIVATIONS = {
    "relu": F.relu,
}

POOLING = {
    "max": global_max_pool,
    "mean": global_mean_pool,
    "sum": global_add_pool,
}

CONVS = {
    "gat": GATConv,
    "gcn": GCNConv,
}

class Rgnn(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes,
                 model_type="gat",
                 pooling="max",
                 activation="relu",
                 num_layers=5):
        super().__init__()

        torch.manual_seed(42)

        Conv = CONVS[model_type]
        self.activation = ACTIVATIONS[activation]
        self.pool = POOLING[pooling]

        self.convs = torch.nn.ModuleList()
        self.convs.append(Conv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(Conv(hidden_channels, hidden_channels))

        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))

        x = self.pool(x, batch)
        return self.lin(x)