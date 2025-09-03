import os
from pathlib import Path

import openml

data_dir = os.environ.get("DATA_DIR")
if data_dir is None:
    data_dir = (Path.cwd() / ".data").resolve()
else:
    data_dir = Path(data_dir).resolve()
DATA_DIR = data_dir


openml_api_key = os.environ.get("OPENML_API_KEY")
if openml_api_key is None:
    raise ValueError("OPENML_API_KEY not found in environment variables.")
openml.config.apikey = openml_api_key

openml_cache_dir = os.environ.get("OPENML_CACHE_DIR")
if openml_cache_dir is None:
    raise ValueError("OPENML_CACHE_DIR not found in environment variables.")
openml.config.set_root_cache_directory(openml_cache_dir)
