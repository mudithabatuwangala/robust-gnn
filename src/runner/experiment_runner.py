from src.utils.config_validator import validate_config
from src.utils.config_normalizer import normalize_config
from experiments.run_experiment import run

def execute(config):
    config = normalize_config(config)
    validate_config(config)
    return run(config)