import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
#create first graph 
x1 = torch.tensor([
    [1,0,1],   # node 0
    [0,1,0],   # node 1
    [1,1,0]    # node 2
], dtype=torch.float)

edge_index1 = torch.tensor([
    [0,1,1,2],
    [1,0,2,1]
], dtype=torch.long)

y1 = torch.tensor([0])
g1 = Data(x=x1, edge_index=edge_index1, y=y1)

# creater graph 2
x2 = torch.tensor([
    [0,1,1],
    [1,0,0],
    [1,1,1]
], dtype=torch.float)

edge_index2 = torch.tensor([
    [0,1,0,2],
    [1,0,2,0]
], dtype=torch.long)

y2 = torch.tensor([1])

g2 = Data(x=x2, edge_index=edge_index2, y=y2)

#create graph 3

x3 = torch.tensor([
    [1,1,0],
    [0,0,1],
    [1,0,1],
    [0,1,0]
], dtype=torch.float)

edge_index3 = torch.tensor([
    [0,1,1,2,2,3],
    [1,0,2,1,3,2]
], dtype=torch.long)

y3 = torch.tensor([0])

g3 = Data(x=x3, edge_index=edge_index3, y=y3)

# batch all three graphs pyg will make one big graph and batch vector
dataset = [g1, g2, g3]
loader = DataLoader(dataset, batch_size=3)

#model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 8)
        self.conv2 = GCNConv(8, 8)
        self.lin = Linear(8, 2)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        print("After Conv:", x.shape)

        x = global_mean_pool(x, batch)

        print("After Pool:", x.shape)

        x = self.lin(x)

        return x
    
    #train
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
#epoch
for epoch in range(20):
    for data in loader:

        print("Total Nodes:", data.x.shape)
        print("Batch vector:", data.batch)

        out = model(data.x, data.edge_index, data.batch)

        print("Predictions:", out)

        loss = criterion(out, data.y)
        print("Loss:", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()