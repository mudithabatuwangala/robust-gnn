import torch

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0

    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)