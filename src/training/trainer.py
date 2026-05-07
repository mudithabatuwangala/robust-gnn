def train(model, loader, optimizer, criterion):
    # model = Rgnn(hidden_channels=128)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)