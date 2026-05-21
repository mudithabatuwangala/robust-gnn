import os
import pandas as pd


def save_or_update_results(new_df, save_path):

    # --------------------------------------------------
    # CASE 1: File does not exist
    # --------------------------------------------------
    if not os.path.exists(save_path):

        new_df.to_csv(save_path, index=False)
        print(f"Created new results file: {save_path}")
        return

    # --------------------------------------------------
    # CASE 2: Load existing file
    # --------------------------------------------------
    existing_df = pd.read_csv(save_path)

    # --------------------------------------------------
    # Define experiment identity columns
    # (what makes an experiment UNIQUE)
    # --------------------------------------------------
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


    # --------------------------------------------------
    # Remove rows that already exist in new_df
    # --------------------------------------------------
    for _, new_row in new_df.iterrows():

        condition = True

        for col in key_cols:
            condition = condition & (existing_df[col] == new_row[col])

        # remove old matching row
        existing_df = existing_df[~condition]

    # --------------------------------------------------
    # Append new rows
    # --------------------------------------------------
    updated_df = pd.concat(
        [existing_df, new_df],
        ignore_index=True
    )

    # --------------------------------------------------
    # Save updated file
    # --------------------------------------------------
    updated_df.to_csv(save_path, index=False)

    print(f"Updated results file: {save_path}")









# import os
# import pandas as pd


# BASELINE_FILE = "results/baseline_results.csv"
# GENERAL_FILE = "results/generalization_results.csv"


# def get_key_cols(experiment_type):

#     if experiment_type == "generalization":
#         return ["dataset", "model", "pooling", "activation"]

#     elif experiment_type == "baseline":
#         return ["dataset", "model", "pooling", "activation",
#                 "hidden_channels", "num_layers", "seeds"]

#     else:
#         raise ValueError("Unknown experiment type")


# def save_or_update_results(new_df):

#     exp_type = new_df["experiment_type"].iloc[0]

#     save_path = BASELINE_FILE if exp_type == "baseline" else GENERAL_FILE

#     key_cols = get_key_cols(exp_type)

#     # -------------------------
#     # create file
#     # -------------------------
#     if not os.path.exists(save_path):
#         new_df.to_csv(save_path, index=False, float_format="%.4f")
#         print(f"Created {save_path}")
#         return

#     existing_df = pd.read_csv(save_path)

#     # -------------------------
#     # remove duplicates
#     # -------------------------
#     for _, row in new_df.iterrows():
#         cond = True
#         for col in key_cols:
#             cond = cond & (existing_df[col] == row[col])
#         existing_df = existing_df[~cond]

#     updated = pd.concat([existing_df, new_df], ignore_index=True)

#     # IMPORTANT: 4 decimal precision
#     updated.to_csv(save_path, index=False, float_format="%.4f")

#     print(f"Updated {save_path}")









# import os
# import pandas as pd


# def save_or_update_results(new_df, save_path):

#     if not os.path.exists(save_path):
#         new_df.to_csv(save_path, index=False)
#         print(f"Created new results file: {save_path}")
#         return

#     existing_df = pd.read_csv(save_path)

#     # -------------------------
#     # AUTO DETECT KEY COLUMNS
#     # -------------------------
#     if "hidden_channels" in new_df.columns:
#         key_cols = ["dataset", "model", "hidden_channels", "num_layers", "seeds"]
#     else:
#         key_cols = ["dataset", "model", "pooling", "activation"]

#     # -------------------------
#     # REMOVE DUPLICATES
#     # -------------------------
#     for _, new_row in new_df.iterrows():

#         condition = True
#         for col in key_cols:
#             condition = condition & (existing_df[col] == new_row[col])

#         existing_df = existing_df[~condition]

#     updated_df = pd.concat([existing_df, new_df], ignore_index=True)
#     updated_df.to_csv(save_path, index=False)

#     print(f"Updated results file: {save_path}")