import yaml
from typing import Dict
import os

def get_config(config_filename = None, approach = None):
    """
    Load base config and override with another config
    
    Args:
        config_filename: Path to specific YAML config
        approach: which approach to use 'dml', 'ttt', 'std_ner'
    
    Returns:
        Merged configuration dictionary or just base config if no config_filename given.
    """
    base_config_path = "./configs/base.yaml"
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    if config_filename:
        override_config_path = os.path.join("./configs", approach, config_filename)
        print(f"Overriding base config with {override_config_path}")
        with open(override_config_path, 'r') as f:
            override_config = yaml.safe_load(f)
        config = _deep_update(config, override_config)
    return config

def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Recursively update a dictionary"""
    for key, value in update_dict.items():
        if (key in base_dict and 
            isinstance(base_dict[key], dict) and 
            isinstance(value, dict)):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict