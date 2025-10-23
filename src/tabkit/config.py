import os
from pathlib import Path

data_dir = os.environ.get("DATA_DIR")
if data_dir is None:
    data_dir = (Path.cwd() / ".data").resolve()
else:
    data_dir = Path(data_dir).resolve()
DATA_DIR = data_dir
