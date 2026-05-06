from collections import defaultdict


def collect_results(all_runs):

    # 4-level nested dict
    data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )

    for config, result in all_runs:

        dataset = config["dataset_name"]
        model = config["model_type"]
        pool = config["pooling"]
        act = config["activation"]

        data[dataset][model][pool][act].append(result)

    return data