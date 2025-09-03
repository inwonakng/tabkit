import copy
import hashlib
import json  # For from_json
from dataclasses import (
    asdict,
    dataclass,
    field,  # Import fields for replace
    fields,
)
from pathlib import Path
from typing import Any  # Make sure Literal is imported if used by subclasses

import yaml


def jsonify(data):
    if isinstance(data, (list, tuple)):
        return [jsonify(item) for item in data]
    if isinstance(data, dict):
        return {k: jsonify(v) for k, v in data.items()}
    if hasattr(data, "tolist"):
        return data.tolist()
    # Handles numpy scalar types like np.float32, np.int64
    if hasattr(data, "item"):
        return data.item()
    return data


@dataclass(kw_only=True)
class Configuration:
    """
    Base class for all configuration. The name of the configuration is
    "default". When loading from a file, the name is set as the configuration
    name unless 'config_name' is specified within the file.
    """

    # this is the readable name of the config we can refer to
    config_name: str = field(default="default")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Automatically make subclasses dataclasses with kw_only=True
        dataclass(kw_only=True)(cls)

    def get_unique_name(self, truncate: int = 8) -> str:
        """Return the configuration name (not unique) and hash (unique)"""
        return self.config_name + "-" + self.hash[:truncate]

    # this is the unique hash of the config content
    @property
    def hash(self) -> str:
        """Return the hash of the configuration content."""
        return self.get_content_hash()

    @classmethod
    def _load_data_and_determine_name(cls, path: str | Path, loader_func):
        path = Path(path)
        with open(path, "r") as f:
            data = loader_func(f)

        # Determine config_name:
        # 1. Use 'config_name' from file content if present.
        # 2. Else, use the filename stem.
        if "config_name" in data:
            # If config_name is in the data, it's part of the **data to be loaded.
            # The instance's config_name will be set from this.
            pass
        else:
            # If not in data, use filename stem and inject it.
            data["config_name"] = path.stem
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path):
        """
        Load the configuration from a file.
        The file can be YAML or JSON. The config_name is derived from the filename
        if not present in the file content.
        """
        path = Path(path)
        if path.suffix == ".yaml" or path.suffix == ".yml":
            return cls.from_yaml(path)
        elif path.suffix == ".json":
            return cls.from_json(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    @classmethod
    def from_yaml(cls, path: str | Path):
        """Load the YAML content from `path` and instantiate the dataclass."""
        return cls._load_data_and_determine_name(path, yaml.safe_load)

    @classmethod
    def from_json(cls, path: str | Path):
        """Load the JSON content from `path` and instantiate the dataclass."""
        return cls._load_data_and_determine_name(path, json.load)  # Use json.load

    @classmethod
    def from_dict(cls, data: dict, config_name: str | None = None):
        """Load the content from python dict and instantiate the dataclass."""
        data_copy = data.copy()  # Avoid modifying original dict
        if config_name:
            data_copy["config_name"] = config_name  # Explicit name overrides
        if "config_name" not in data_copy:
            # If still no config_name, raise error or assign a default
            # For consistency with file loading (where name is derived if not present),
            # one might argue for a default here too, or require it.
            # Your original code raised an error, which is a valid choice.
            raise ValueError(
                "No 'config_name' provided in data or as an argument for the configuration object!"
            )
        return cls(**data_copy)

    def copy(self):
        """Return a deep copy of the configuration instance."""
        return copy.deepcopy(self)

    def update(self, overrides: dict[str, Any] | None = None, **kwargs):
        """
        Update the configuration instance in-place.
        Note: This modifies the current instance. For an immutable update, use `replace()`.
        """
        # Ensure all keys are valid fields of the dataclass
        valid_fields = {f.name for f in fields(self)}

        if overrides is not None:
            for key, value in overrides.items():
                if key in valid_fields:
                    setattr(self, key, value)
                else:
                    # Optionally raise an error or warning for unknown keys
                    print(
                        f"Warning: '{key}' is not a valid configuration field. Skipping."
                    )

        for key, value in kwargs.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                print(f"Warning: '{key}' is not a valid configuration field. Skipping.")
        return self

    def replace(self, **changes):
        """
        Return a new configuration instance with specified fields replaced.
        Similar to `dataclasses.replace`. The `config_name` is copied by default
        but can also be changed via the `changes` argument.
        """
        # `dataclasses.replace` doesn't work directly if `__init__` is custom
        # or if we want to ensure our kw_only behavior.
        # A simple way is to convert to dict, update, and create new.
        current_as_dict = asdict(self)
        current_as_dict.update(changes)
        return type(self)(**current_as_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        Excludes fields that are None and 'config_name'.
        For general serialization (e.g. to YAML), 'config_name' should be included
        if it was explicitly set or loaded.
        """
        return {
            k: v
            for k, v in asdict(self).items()
            if v is not None
            # Always exclude config_name from this representation
            and k != "config_name"
        }

    def get_content_hash(self, HASH_METHOD=hashlib.sha256) -> str:
        """
        Computes a hash of the configuration's content (excluding 'config_name').
        Ensures a canonical representation for hashing.
        """
        # Use self.to_dict() which already excludes config_name and None values
        data_to_hash = self.to_dict()

        # Serialize to a canonical JSON string (sorted keys)
        # Using str(v) for all values for simplicity, but consider specific types if needed
        # e.g. float precision. For most configs, str() is fine.
        canonical_json = json.dumps(
            jsonify(data_to_hash), sort_keys=True, separators=(",", ":")
        )

        hasher = HASH_METHOD()
        hasher.update(canonical_json.encode("utf-8"))
        return hasher.hexdigest()

    def save_yaml(self, path: str | Path):
        """Save the configuration content (excluding config_name) to a YAML file."""
        # self.to_dict() already excludes config_name.
        # This means config_name is never written to the file.
        # On loading, config_name will be derived from the filename.
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(path, "w") as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                sort_keys=False,
                allow_unicode=True,
            )
        return self

    def save_json(self, path: str | Path):
        """Save the configuration content (excluding config_name) to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(path, "w") as f:
            json.dump(
                self.to_dict(),
                f,
                sort_keys=False,
                indent=2,
            )  # indent for readability
        return self
