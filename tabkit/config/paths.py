from pathlib import Path
import os
import dotenv
import openml


ROOT_DIR = Path(__file__).parent.parent.parent

dotenv.load_dotenv(ROOT_DIR / ".env")

data_dir = os.environ.get("DATA_DIR")
if data_dir is None:
    data_dir = (ROOT_DIR / ".data").resolve()
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

HF_API_KEY = os.environ.get("HF_API_KEY")
if HF_API_KEY is None:
    raise ValueError("HF_API_KEY not found in environment variables.")

HF_HOME = os.environ.get("HF_HOME")
if HF_HOME is None:
    raise ValueError("HF_HOME not found in environment variables")

REPORT_DIR = os.environ.get("REPORT_DIR")
if REPORT_DIR is None:
    raise ValueError("REPORT_DIR not found in environment variables")
