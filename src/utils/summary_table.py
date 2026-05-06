import pandas as pd

def build_table(grouped):

    rows = []

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

                        "best_val_acc": sum(r["best_val_acc"] for r in runs) / len(runs),
                        "challenge_acc": sum(r["challenge_acc"] for r in runs) / len(runs),
                        "min_loss": sum(r["min_loss"] for r in runs) / len(runs),
                        # "early_stop_epoch": sum(r["early_stop_epoch"] for r in runs) / len(runs),
                    })

    return pd.DataFrame(rows)