def split_by_num_nodes(dataset, threshold):
    train_val = [d for d in dataset if d.num_nodes <= threshold]
    challenge = [d for d in dataset if d.num_nodes > threshold]
    return train_val, challenge