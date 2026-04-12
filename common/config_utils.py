"""
Configuration utilities for the MRI reconstruction framework.
"""

import json
import os
from pathlib import Path

def load_config(config_path):
    """
    Load configuration from JSON file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def merge_configs(base_config, specific_config):
    """
    Merge base configuration with specific configuration.
    Specific config values override base config values.

    Args:
        base_config (dict): Base configuration
        specific_config (dict): Specific configuration

    Returns:
        dict: Merged configuration
    """
    merged = base_config.copy()

    def deep_merge(base, specific):
        for key, value in specific.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value

    deep_merge(merged, specific_config)
    return merged

def load_config_with_base(config_path, base_config_path=None):
    """
    Load configuration with optional base config inheritance.

    Args:
        config_path (str): Path to the specific configuration file
        base_config_path (str, optional): Path to the base configuration file

    Returns:
        dict: Merged configuration dictionary
    """
    config = load_config(config_path)

    if base_config_path and os.path.exists(base_config_path):
        base_config = load_config(base_config_path)
        config = merge_configs(base_config, config)

    return config

def save_config(config, save_path):
    """
    Save configuration to JSON file.

    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)