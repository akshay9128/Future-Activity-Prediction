"""
Configuration loader utility.
Loads YAML configuration files and returns a dictionary.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: Path | str = "configs/default_config.yaml") -> Dict[str, Any]:
    """
    Safely load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration options.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file contains invalid syntax.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must parse into a dictionary.")

    return config
