import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# @torch.no_grad()
# def evaluate(model, loader):
#     model.eval()
#     correct = 0

#     for data in loader:
#         out = model(data.x, data.edge_index, data.batch)
#         pred = out.argmax(dim=1)
#         correct += int((pred == data.y).sum())

#     return correct / len(loader.dataset)

# @torch.no_grad()
# def evaluate_metrics(model, loader):

#     model.eval()

#     all_preds = []
#     all_labels = []

#     for data in loader:

#         out = model(data.x, data.edge_index, data.batch)

#         pred = out.argmax(dim=1)

#         y = data.y.view(-1).long()

#         all_preds.extend(pred.cpu().numpy())
#         all_labels.extend(y.cpu().numpy())

#     precision = precision_score(
#         all_labels,
#         all_preds,
#         average="binary",
#         zero_division=0
#     )

#     recall = recall_score(
#         all_labels,
#         all_preds,
#         average="binary",
#         zero_division=0
#     )

#     f1 = f1_score(
#         all_labels,
#         all_preds,
#         average="binary",
#         zero_division=0
#     )

#     return precision, recall, f1

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []

    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        y = data.y.view(-1).long()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        accuracy = correct / len(loader.dataset)

        precision = precision_score(
            all_labels,
            all_preds,
            average="binary",
            zero_division=0
        )

        recall = recall_score(
            all_labels,
            all_preds,
            average="binary",
            zero_division=0
        )

        f1 = f1_score(
            all_labels,
            all_preds,
            average="binary",
            zero_division=0
        )

    return accuracy, precision, recall, f1