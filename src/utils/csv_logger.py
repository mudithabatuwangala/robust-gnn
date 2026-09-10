import csv
import os

def log_results_csv(results, path="results/experiments.csv"):

    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "dataset",
                "model",
                "activation",
                "pooling",
                "num_layers",
                "hidden_channels",
                "seed",
                "best_val_acc",
                "challenge_acc",
                "min_loss"
            ])

        for r in results:

            cfg = r["config"]
            met = r["metrics"]

            writer.writerow([
                cfg["dataset"],
                cfg["model"],
                cfg["activation"],
                cfg["pooling"],
                cfg["num_layers"],
                cfg["hidden_channels"],
                cfg["seed"],
                round(met["best_val_acc"], 4),
                round(met["challenge_acc"], 4),
                round(met["min_loss"], 4),
            ])