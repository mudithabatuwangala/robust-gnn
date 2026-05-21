def run_multiple_times(run_fn, config, num_runs=5):
    results = []

    # for i in range(num_runs):
    #     print(f"\n===== Run {i+1}/{num_runs} =====")
    #     result = run_fn(config)
    #     results.append(result)

    for i in range(num_runs):
        print(f"\n===== Run {i+1}/{num_runs} =====")
        # config["seeds"] = config.get("seeds", 42) + i  # vary seed
        result = run_fn(config)
        results.append(result)

    return results