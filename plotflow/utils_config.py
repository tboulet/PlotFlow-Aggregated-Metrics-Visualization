import os
import sys

# Constants for the config path and default config name
DIR_CONFIGS: str = "configs_plotflow"
NAME_CONFIG_DEFAULT: str = "config_default.yaml"


def assert_config_dir_exists():
    """Check if the configs_plotflow directory exists."""
    if os.path.exists(DIR_CONFIGS) and os.path.isdir(DIR_CONFIGS):
        return
    else:
        print(f"Error: Config directory '{DIR_CONFIGS}' does not exist.")
        print("To initialize the config, please run: 'plotflow-init-config' before.")
        print(
            "Then you can run 'plotflow' with eventual Hydra overrides (config name, config overrides)."
        )
        print("See the README for more information.")
        sys.exit(1)
