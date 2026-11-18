import importlib
from functools import lru_cache
from typing import Dict

from torch import nn

# Model path mapping
_MODEL_PATHS: Dict[str, str] = {
    "SURMT": ".SURMT.Networks"
}


@lru_cache(maxsize=None)
def _import_class(model_name: str) -> nn.Module:
    """
    Dynamically import model class

    Args:
        model_name: Model name

    Returns:
        model_class: Model class

    Raises:
        ValueError: When model is not registered or imported failed
    """
    try:
        module_path = _MODEL_PATHS.get(model_name)
        if not module_path:
            available_models = ", ".join(list(_MODEL_PATHS.keys())[:10])
            raise ValueError(
                f"Model '{model_name}' is not registered.\n"
                f"Available models: {available_models}...\n"
                f"Total {len(_MODEL_PATHS)} models available."
            )

        module = importlib.import_module(module_path, package=__package__)
        return getattr(module, model_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Cannot import model {model_name}: {str(e)}")


def get_model(name: str) -> nn.Module:
    """
    Get model class by name

    Args:
        name: Model name

    Returns:
        model_class: Model class
    """
    return _import_class(name)


def get_available_models() -> list:
    """
    Get all available model names

    Returns:
        model_names: Model names list
    """
    return list(_MODEL_PATHS.keys())


__all__ = list(_MODEL_PATHS.keys()) + [
    "get_model",
    "get_available_models"
]
