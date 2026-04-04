import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes):
        super(GCN, self).__init__()
        # Three layers of 'message passing'
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # The final decision-maker
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Talk to neighbors
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)

        # 2. Summary: Squash all nodes into one graph representation
        x = global_mean_pool(x, batch) 

        # 3. Predict: Use Dropout to prevent memorization
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)