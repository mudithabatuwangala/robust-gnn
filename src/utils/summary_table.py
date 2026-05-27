import pandas as pd

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
                            "small_acc": round(sum(r["small_acc"] for r in runs) / len(runs), 4),
                            "large_acc": round(sum(r["large_acc"] for r in runs) / len(runs), 4),
                            "min_loss": round(sum(r["min_loss"] for r in runs) / len(runs), 4),
                            # "early_stop_epoch": round(sum(r["early_stop_epoch"] for r in runs) / len(runs), 4),
                            "small_precision": round(sum(r["small_precision"] for r in runs) / len(runs), 4),
                            "small_recall": round(sum(r["small_recall"] for r in runs) / len(runs), 4),
                            "small_f1_score": round(sum(r["small_f1_score"] for r in runs) / len(runs), 4),
                            "large_precision": round(sum(r["large_precision"] for r in runs) / len(runs), 4),
                            "large_recall": round(sum(r["large_recall"] for r in runs) / len(runs), 4),
                            "large_f1_score": round(sum(r["large_f1_score"] for r in runs) / len(runs), 4),
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
                        "small_acc": round(sum(r["small_acc"] for r in runs) / len(runs), 4),
                        "large_acc": round(sum(r["large_acc"] for r in runs) / len(runs), 4),
                        "min_loss": round(sum(r["min_loss"] for r in runs) / len(runs), 4),
                        # "early_stop_epoch": round(sum(r["early_stop_epoch"] for r in runs) / len(runs), 4),
                        "small_precision": round(sum(r["small_precision"] for r in runs) / len(runs), 4),
                        "small_recall": round(sum(r["small_recall"] for r in runs) / len(runs), 4),
                        "small_f1_score": round(sum(r["small_f1_score"] for r in runs) / len(runs), 4),
                        "large_precision": round(sum(r["large_precision"] for r in runs) / len(runs), 4),
                        "large_recall": round(sum(r["large_recall"] for r in runs) / len(runs), 4),
                        "large_f1_score": round(sum(r["large_f1_score"] for r in runs) / len(runs), 4),
                    })
    return pd.DataFrame(rows)