import os
import pandas as pd

def save_or_update_results(new_df, save_path):
    
    if not os.path.exists(save_path):
        new_df.to_csv(save_path, index=False)
        print(f"Created new results file: {save_path}")
        return

    existing_df = pd.read_csv(save_path)

    if save_path == "results/generalization_experiment":
        key_cols = [
            "dataset",
            "model",
            "pooling",
            "activation"
        ]
    else:
        key_cols = [
            "seeds",
            "num_layers",
            "num_channels",
        ]

    # Remove rows that already exist in new_df
    for _, new_row in new_df.iterrows():
        condition = True
        for col in key_cols:
            condition = condition & (existing_df[col] == new_row[col])
        # remove old matching row
        existing_df = existing_df[~condition]
    # Append new rows
    updated_df = pd.concat(
        [existing_df, new_df],
        ignore_index=True
    )

    updated_df.to_csv(save_path, index=False)
    print(f"Updated results file: {save_path}")