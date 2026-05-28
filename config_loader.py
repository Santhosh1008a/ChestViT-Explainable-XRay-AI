"""
config_loader.py
----------------
Loads config/config.yaml and exposes a typed namespace for the rest of the project.
"""
import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert dict → SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_config(config_path: str | Path | None = None) -> SimpleNamespace:
    """
    Load the project config from YAML.

    Args:
        config_path: Path to config.yaml. Defaults to <project_root>/config/config.yaml.

    Returns:
        SimpleNamespace with dot-access to all config values.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dict_to_namespace(raw)


# Convenience singleton — import this anywhere in the project
CFG = load_config()
