import pandas as pd


def build_baseline_table(all_runs):

    rows = []

    for cfg, result in all_runs:

        rows.append({
            "dataset": cfg["dataset_name"],
            "model": cfg["model_type"],
            "hidden_channels": cfg["hidden_channels"],
            "num_layers": cfg["num_layers"],
            "seeds": cfg["seeds"],

            "best_val_acc": round(result["best_val_acc"], 4),
            "challenge_acc": round(result["challenge_acc"], 4),
            "min_loss": round(result["min_loss"], 4),
        })

    return pd.DataFrame(rows)