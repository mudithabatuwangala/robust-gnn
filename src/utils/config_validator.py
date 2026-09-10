def validate_config(config):
    allowed_models = ["gat", "gcn", "gin"]
    allowed_activations = ["relu", "elu", "leaky_relu"]
    allowed_pooling = ["max", "mean", "sum"]

    assert config["model_type"] in allowed_models, "Invalid model_type"
    assert config["activation"] in allowed_activations, "Invalid activation"
    assert config["pooling"] in allowed_pooling, "Invalid pooling"

    assert config["hidden_channels"] > 0
    assert config["num_layers"] > 0
    assert 0 < config["train_ratio"] < 1