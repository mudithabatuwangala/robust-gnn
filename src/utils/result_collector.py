from collections import defaultdict


# def collect_results(all_runs):

#     # 4-level nested dict
#     data = defaultdict(
#         lambda: defaultdict(
#             lambda: defaultdict(
#                 lambda: defaultdict(list)
#             )
#         )
#     )

#     for config, result in all_runs:

#         dataset = config["dataset_name"]
#         model = config["model_type"]
#         pool = config["pooling"]
#         act = config["activation"]

#         data[dataset][model][pool][act].append(result)

#     return data




# from collections import defaultdict


# def collect_results(all_runs):

#     data = defaultdict(
#         lambda: defaultdict(
#             lambda: defaultdict(
#                 lambda: defaultdict(
#                     lambda: defaultdict(list)
#                 )
#             )
#         )
#     )

#     for config, result in all_runs:

#         exp_type = config["experiment_type"]

#         dataset = config["dataset_name"]
#         model = config["model_type"]
#         pool = config["pooling"]
#         act = config["activation"]

#         data[exp_type][dataset][model][pool][act].append(result)

#     return data





def collect_results(all_runs, exp_type):

    if exp_type == "generalization":
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
    else:
        # 3-level nested dict
        data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
        for config, result in all_runs:
            seeds = config["seeds"]
            num_layers = config["num_layers"]
            num_channels = config["hidden_channels"]
            data[seeds][num_layers][num_channels].append(result)
    return data