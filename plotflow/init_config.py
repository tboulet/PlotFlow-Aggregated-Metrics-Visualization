import os
import shutil
from pathlib import Path

# Import path constants for config management
from plotflow.utils_config import DIR_CONFIGS, NAME_CONFIG_DEFAULT


def init_config(
    dir_configs: str = DIR_CONFIGS, name_config_default: str = NAME_CONFIG_DEFAULT
):
    """Initializes the default config in the current repository.

    Args:
        dir_configs (str): The directory where the default config should be placed, e.g. 'configs_plotflow'.
        name_config_default (str): The name (with .yaml extension) of the default config file, e.g. 'config_default.yaml'.
    """
    config_dir = Path(
        dir_configs
    )  # The location where the default config should be placed
    config_dir.mkdir(parents=True, exist_ok=True)

    # Define the path to the default config
    default_config_path = Path(__file__).parent / f"{dir_configs}/{name_config_default}"

    # Ensure default_config.yaml is in place
    default_config_dest = config_dir / name_config_default
    if not default_config_dest.exists():
        shutil.copy(default_config_path, default_config_dest)
        print(f"Default config created at {default_config_dest}")
    else:
        print(
            f"Default config already exists at {default_config_dest}. Please remove/rename it and run this script again to recreate it."
        )


def main():
    init_config()


if __name__ == "__main__":
    main()
