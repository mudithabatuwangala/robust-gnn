from src.utils.config_validator import validate_config
from src.utils.config_normalizer import normalize_config
from experiments.run_experiment import run


def execute(config):

    # 1. Normalize (UI → internal)
    config = normalize_config(config)

    # 2. Validate
    validate_config(config)

    # 3. Run and RETURN result ✅
    return run(config)