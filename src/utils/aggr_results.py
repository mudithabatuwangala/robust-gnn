def aggregate_results(results):
    aggregated = {}

    for key in results[0].keys():
        aggregated[key] = [r[key] for r in results]

    return aggregated