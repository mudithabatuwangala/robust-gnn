import argparse
import yaml
import argparse
import yaml
import itertools
import copy
from src.runner.repeat_runner import run_multiple_times
from src.runner.experiment_runner import execute
from src.utils.result_collector import collect_results
from src.utils.summary_table import build_table
from src.utils.advanced_plotting import plot_dataset_results
from src.utils.save_results import save_or_update_results


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(defaults, override):
    config = defaults.copy()
    config.update(override)
    return config


def generate_configs(config):
    keys = []
    values = []

    for k, v in config.items():
        if isinstance(v, list):
            keys.append(k)
            values.append(v)

    if not keys:
        return [config]

    combinations = list(itertools.product(*values))
    configs = []

    for combo in combinations:
        new_config = copy.deepcopy(config)
        for i, key in enumerate(keys):
            new_config[key] = combo[i]
        configs.append(new_config)

    return configs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configs
    global_config = load_yaml("configs/defaults/global.yaml")
    exp_config = load_yaml(args.config)
    merged_config = merge_configs(global_config, exp_config)
    configs = generate_configs(merged_config)

    print(f"\nRunning {len(configs)} experiment configs...\n")

    all_runs = []
    # Run experiments
    for i, cfg in enumerate(configs):

        print(f"\n===== Experiment {i+1}/{len(configs)} =====")
        print(cfg)

        run_results = run_multiple_times(execute, cfg, num_runs=5)

        # store all runs with config
        for r in run_results:
            all_runs.append((cfg, r))

    grouped = collect_results(all_runs, cfg["experiment_type"])
    df = build_table(grouped, cfg["experiment_type"])
    path = "results/" + str(cfg["experiment_type"]) + "_experiment.csv"
    save_or_update_results(df, path)

    print("\nSaved summary table to " + path)