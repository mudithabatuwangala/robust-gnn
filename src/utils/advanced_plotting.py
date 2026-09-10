# import matplotlib.pyplot as plt


# def plot_boxplots(metrics_dict, title="Experiment Results"):

#     keys = list(metrics_dict.keys())
#     values = [metrics_dict[k] for k in keys]

#     plt.figure()

#     plt.boxplot(values)
#     plt.xticks(range(1, len(keys) + 1), keys, rotation=30)

#     plt.title(title)
#     plt.ylabel("Value")

#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def plot_dataset_results(dataset_name, grouped):

    models = ["gcn", "gat", "gin"]
    poolings = ["max", "mean", "sum"]
    activations = ["relu", "elu", "leaky_relu"]

    os.makedirs("results/figures", exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    outer = gridspec.GridSpec(
        len(models), 1,
        figure=fig,
        hspace=0.5
    )

    for i, model in enumerate(models):

        model_grid = gridspec.GridSpecFromSubplotSpec(
            1, len(poolings),
            subplot_spec=outer[i],
            wspace=0.3
        )

        for j, pool in enumerate(poolings):

            ax = fig.add_subplot(model_grid[j])

            ax.set_title(f"{model.upper()} | {pool.upper()}")

            activation_data = []

            for act in activations:

                runs = grouped.get(model, {}).get(pool, {}).get(act, [])

                if runs:
                    values = [r["best_val_acc"] for r in runs]
                else:
                    values = []

                activation_data.append(values)

            ax.boxplot(activation_data)
            ax.set_xticklabels(activations, rotation=30)
            ax.set_ylim(0, 1)

            if j == 0:
                ax.set_ylabel(model.upper())

    fig.suptitle(
        f"{dataset_name} - Model × Pooling × Activation Analysis",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = f"results/figures/{dataset_name}_summary.png"
    plt.savefig(save_path, dpi=300)

    print(f"Saved plot: {save_path}")

    plt.show()