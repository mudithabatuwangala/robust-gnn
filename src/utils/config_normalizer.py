def normalize_config(config):
    mapping = {
        "GAT": "gat",
        "GCN": "gcn",
        "ReLU": "relu",
        "ELU": "elu",
    }

    for key, value in config.items():
        if isinstance(value, list):
            config[key] = [mapping.get(v, v).lower() for v in value]
        elif isinstance(value, str):
            config[key] = mapping.get(value, value).lower()

    return config