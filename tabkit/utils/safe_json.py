from typing import Any
import re
import json
from pathlib import Path
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


def safe_dump(value: dict, file: str | Path) -> None:
    """Converts a dictionary to a json serializable dictionary and writes it to a file.

    Args:
        `value` (`dict`): The dictionary to be written to a file.
        `file` (`str | Path`): Name of the file to write the dictionary to. Can be a string or a Path object.

    Raises:
        `Exception`: If there is an error writing to the file or converting the dictionary to a json serializable dictionary.
    """
    try:
        with open(file, "w") as f:
            json.dump(safe_json(value), f, indent=2)
    except Exception as e:
        raise Exception(f"Error dumping {file}: {str(e)}")


def safe_load(file: str | Path) -> dict:
    """Loads a json file and returns it as a dictionary.

    Args:
        `file` (`str | Path`): Name of the file to load. Can be a string or a Path object.

    Raises:
        `Exception`: If there is an error loading the file.

    Returns:
        `dict`
        The contents of the file as a dictionary.
    """
    try:
        with open(file) as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error loading {file}: {str(e)}")


def safe_str(input_string: str) -> str:
    """
    Converts the given string to a Unix-safe form by replacing
    all non-alphanumeric and non-underscore characters with underscores.
    """
    # Replace anything not a-z, A-Z, 0-9, or underscore with '_'
    return re.sub(r"[^a-zA-Z0-9_]", "_", input_string.replace("/","__"))
