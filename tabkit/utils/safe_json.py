from typing import Any

import numpy as np
import torch


def safe_json(
    value: dict[str, Any] | Any,
) -> dict:
    """Turns a python dictionary to a json serializable dictionary.

    Args:
        `value` (`dict[str, Any] | any`): Some dictionary to be converted to a json serializable dictionary. The value
        must either be numpy or torch tensors or native python types.

    Raises:
        `ValueError`: If the value is not a numpy or torch tensor or a native python type.

    Returns:
        `dict`
        Returns a dictionary that is json serializable.
    """
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [safe_json(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(safe_json(v) for v in value)
    elif isinstance(value, (np.ndarray, torch.Tensor)):
        return [v.item() for v in value.tolist()]
    elif isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    elif isinstance(value, np.generic):
        return value.item()
    else:
        if not isinstance(value, (int, float, str, bool)):
            raise ValueError(f"Unsupported type: {type(value)}")
        return value
