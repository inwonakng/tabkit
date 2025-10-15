# Table Toolkit (tabkit)

A python library for consistent preprocessing of tabular data. Handles column
type inference, missing value imputation, feature binning, stratified
split/sampling and more in a configuration-driven manner. I made this toolkit because I needed a way to reliably preprocess/cache datasets in a reproducible manner.

## Installation

Stable release via PyPI:

```bash
pip install table-toolkit
```

Or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/inwonakng/tabkit.git@main
```

This package has been tested only with Python 3.10 and above.

## Quick Start

```python
from tabkit import TableProcessor, DatasetConfig, TableProcessorConfig

# Define your dataset and processing configs
dataset_config = DatasetConfig(
    dataset_name="my_dataset",
    data_source="disk",
    file_path="path/to/your/data.csv",
    file_type="csv",
    label_col="target"
)

processor_config = TableProcessorConfig(
    task_kind="classification",  # or "regression"
    n_splits=5,
    random_state=42
)

# Create processor
processor = TableProcessor(
    dataset_config=dataset_config,
    config=processor_config
)

# Prepare data (this caches results for future runs)
processor.prepare()

# Get splits
X_train, y_train = processor.get_split("train")
X_val, y_val = processor.get_split("val")
X_test, y_test = processor.get_split("test")

# Or get the raw dataframe
df = processor.get("raw_df")
```

**Note:** You can also use plain dictionaries instead of config classes - both work identically! See [Configuration Options](#configuration-options) below.

For more examples, see [examples/basic_usage.py](examples/basic_usage.py).

## Features

- **Automatic type inference**: Detects categorical, continuous, binary, and datetime columns
- **Flexible preprocessing pipelines**: Chain transforms like imputation, encoding, scaling, discretization
- **Smart caching**: Preprocessed data is cached based on config hash - perfect for distributed training
- **Stratified splitting**: Automatically handles stratified train/val/test splits
- **Reproducible**: Same config always produces same results

## Configuration Options

Tabkit provides **type-safe configuration classes** with IDE autocomplete and inline documentation. You can also use plain dictionaries if you prefer - both approaches work identically.

### Using Config Classes (Recommended)

```python
from tabkit import DatasetConfig, TableProcessorConfig

# Dataset configuration with type hints and autocomplete
dataset_config = DatasetConfig(
    dataset_name="my_dataset",
    data_source="disk",      # "disk", "openml", "uci", "automm"
    file_path="data.csv",
    file_type="csv",         # "csv" or "parquet"
    label_col="target"
)

# Processor configuration
processor_config = TableProcessorConfig(
    task_kind="classification",  # or "regression"
    random_state=42,
    pipeline=[...],              # Custom pipeline (optional)
    exclude_columns=["id"],      # Columns to exclude (optional)

    # Splitting configuration - see next section
    test_ratio=0.2,              # For ratio-based splitting
    val_ratio=0.1,               # For ratio-based splitting
    # OR
    n_splits=10,                 # For K-fold splitting
    split_idx=0                  # For K-fold splitting
)
```

For detailed documentation on all available options, see the docstrings in `DatasetConfig` and `TableProcessorConfig`, or check the [config source](src/tabkit/data/data_config.py).

### Using Plain Dictionaries (Also supported)

```python
# Same functionality, dictionary-based
dataset_config = {
    "dataset_name": "my_dataset",
    "data_source": "disk",
    "file_path": "data.csv",
    "file_type": "csv",
    "label_col": "target"
}

processor_config = {
    "task_kind": "classification",
    "test_ratio": 0.2,
    "val_ratio": 0.1,
    "random_state": 42
}
```

## Data Splitting Modes

Tabkit supports two distinct approaches for splitting your data into train/validation/test sets. Choose based on your use case:

### Mode 1: Ratio-Based Splitting (Quick & Simple)

**When to use:**
- You want a simple percentage-based split (e.g., 70/15/15)
- You're doing quick prototyping or one-off experiments
- You don't need full dataset coverage

**How it works:**
- Performs a single random stratified split based on specified ratios
- Fast and intuitive
- Different random seeds give different splits, but no systematic coverage

**Example:**
```python
from tabkit import TableProcessorConfig

config = TableProcessorConfig(
    test_ratio=0.2,       # 20% test
    val_ratio=0.1,        # 10% validation
    random_state=42       # 70% training
)
```

### Mode 2: K-Fold Based Splitting (Robust & Reproducible)

**When to use:**
- You need robust cross-validation
- You want to ensure every sample appears in the test set across multiple runs
- You're benchmarking models or doing comprehensive evaluation

**How it works:**
- Uses K-fold cross-validation for systematic data splitting
- By varying `split_idx` from 0 to `n_splits-1`, every sample appears in the test set exactly once
- Provides systematic coverage of your entire dataset
- Default: 10 splits = 10% test, then 9 sub-splits on training portion = ~11% val, ~79% train

**Example:**
```python
from tabkit import TableProcessorConfig

# Run 1: Use fold 0 as test
config = TableProcessorConfig(n_splits=5, split_idx=0)  # 20% test, rest train+val

# Run 2: Use fold 1 as test
config = TableProcessorConfig(n_splits=5, split_idx=1)  # Different 20% test

# ... Run 3-5 to cover all data in test set
```

### Which Mode is Used?

**Priority:** If both `test_ratio` and `val_ratio` are set, ratio-based splitting is used. Otherwise, K-fold splitting is used.

```python
# This uses RATIO mode
config = {"test_ratio": 0.2, "val_ratio": 0.1}

# This uses K-FOLD mode
config = {"n_splits": 10, "split_idx": 0}

# This also uses K-FOLD mode (ratios are None by default)
config = {}  # Uses all defaults
```
