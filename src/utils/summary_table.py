import pandas as pd

# def build_table(grouped):
#     rows = []
#     for dataset in grouped:
#         for model in grouped[dataset]:
#             for pool in grouped[dataset][model]:
#                 for act in grouped[dataset][model][pool]:
#                     runs = grouped[dataset][model][pool][act]
#                     rows.append({
#                         "dataset": dataset,
#                         "model": model,
#                         "pooling": pool,
#                         "activation": act,

#                         "best_val_acc": round(sum(r["best_val_acc"] for r in runs) / len(runs), 4),
#                         "challenge_acc": round(sum(r["challenge_acc"] for r in runs) / len(runs), 4),
#                         "min_loss": round(sum(r["min_loss"] for r in runs) / len(runs), 4),
#                         # "early_stop_epoch": round(sum(r["early_stop_epoch"] for r in runs) / len(runs), 4),
#                     })
#     return pd.DataFrame(rows)


# def build_table(grouped):

#     rows = []

#     for dataset in grouped:
#         for model in grouped[dataset]:
#             for pool in grouped[dataset][model]:
#                 for act in grouped[dataset][model][pool]:

#                     runs = grouped[dataset][model][pool][act]
#                     if len(runs) == 0:
#                         continue

#                     exp_type = runs[0][0].get("experiment_type", "unknown")

#                     row = {
#                         "experiment_type": exp_type,
#                         "dataset": dataset,
#                         "model": model,
#                         "pooling": pool,
#                         "activation": act,

#                         "best_val_acc": round(sum(r["best_val_acc"] for r in runs) / len(runs), 4),
#                         "challenge_acc": round(sum(r["challenge_acc"] for r in runs) / len(runs), 4),
#                         "min_loss": round(sum(r["min_loss"] for r in runs) / len(runs), 4),
#                     }

#                     # baseline-only fields
#                     if exp_type == "baseline":
#                         row.update({
#                             "hidden_channels": runs[0]["hidden_channels"],
#                             "num_layers": runs[0]["num_layers"],
#                             "seeds": runs[0]["seeds"],
#                         })

#                     rows.append(row)

#     return pd.DataFrame(rows)





def build_table(grouped, exp_type):
    rows = []
    if exp_type == "generalization":
        for dataset in grouped:
            for model in grouped[dataset]:
                for pool in grouped[dataset][model]:
                    for act in grouped[dataset][model][pool]:
                        runs = grouped[dataset][model][pool][act]
                        rows.append({
                            "dataset": dataset,
                            "model": model,
                            "pooling": pool,
                            "activation": act,

                            "best_val_acc": round(sum(r["best_val_acc"] for r in runs) / len(runs), 4),
                            "challenge_acc": round(sum(r["challenge_acc"] for r in runs) / len(runs), 4),
                            "min_loss": round(sum(r["min_loss"] for r in runs) / len(runs), 4),
                            # "early_stop_epoch": round(sum(r["early_stop_epoch"] for r in runs) / len(runs), 4),
                            "precision": round(sum(r["precision"] for r in runs) / len(runs), 4),
                            "recall": round(sum(r["recall"] for r in runs) / len(runs), 4),
                            "f1_score": round(sum(r["f1_score"] for r in runs) / len(runs), 4),
                        })
    else:
        for seeds in grouped:
            for num_layers in grouped[seeds]:
                for num_channels in grouped[seeds][num_layers]:
                    runs = grouped[seeds][num_layers][num_channels]
                    rows.append({
                        "seeds": seeds,
                        "num_layers": num_layers,
                        "num_channels": num_channels,

                        "best_val_acc": round(sum(r["best_val_acc"] for r in runs) / len(runs), 4),
                        "challenge_acc": round(sum(r["challenge_acc"] for r in runs) / len(runs), 4),
                        "min_loss": round(sum(r["min_loss"] for r in runs) / len(runs), 4),
                        # "early_stop_epoch": round(sum(r["early_stop_epoch"] for r in runs) / len(runs), 4),
                        "precision": round(sum(r["precision"] for r in runs) / len(runs), 4),
                        "recall": round(sum(r["recall"] for r in runs) / len(runs), 4),
                        "f1_score": round(sum(r["f1_score"] for r in runs) / len(runs), 4),
                    })
    return pd.DataFrame(rows)