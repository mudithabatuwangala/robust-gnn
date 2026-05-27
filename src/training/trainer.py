import torch

def train(model, loader, optimizer, criterion):

    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        # Label handling
        y = data.y
        # correct shape for classification
        if y.dim() > 1:
            y = y.view(-1)
        # CrossEntropy expects long
        y = y.long()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)