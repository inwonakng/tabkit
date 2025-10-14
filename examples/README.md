# Tabkit Examples

This directory contains example scripts demonstrating how to use tabkit.

## Running the Examples

```bash
python examples/basic_usage.py
```

## What's Included

### basic_usage.py

Comprehensive examples covering:

1. **Basic Usage** - Simple classification task with CSV data
2. **Custom Pipeline** - Creating custom preprocessing pipelines for regression
3. **Advanced Config** - Excluding columns, handling dates, multi-class classification
4. **Minimal Config** - Using default settings with minimal configuration
5. **Accessing Data** - Different ways to access raw and processed data

Each example is self-contained and generates its own temporary data.

## Example Structure

Each example demonstrates:
- Creating dataset and processor configurations as dictionaries
- Setting up TableProcessor
- Preparing data (with automatic caching)
- Accessing train/val/test splits
- Working with column metadata
- Customizing preprocessing pipelines

## Key Concepts

### Configuration as Dictionaries

Tabkit uses plain Python dictionaries for configuration:

```python
dataset_config = {
    "dataset_name": "my_data",
    "data_source": "disk",
    "file_path": "data.csv",
    "file_type": "csv",
    "label_col": "target"
}

processor_config = {
    "task_kind": "classification",
    "n_splits": 5,
    "random_state": 42
}
```

### Caching

Tabkit automatically caches preprocessed data based on a hash of your configuration. This means:
- First run: Data is loaded and preprocessed
- Subsequent runs: Data is loaded from cache (much faster!)
- Different configs: Different cache directories

### Custom Pipelines

You can define custom preprocessing pipelines:

```python
pipeline = [
    {"class": "Impute", "params": {"method": "median"}},
    {"class": "Encode", "params": {"method": "most_frequent"}},
    {"class": "Scale", "params": {"method": "standard"}},
]
```

Available transforms:
- **Impute**: Handle missing values (methods: mean, median, most_frequent, constant)
- **Encode**: Encode categorical variables
- **Scale**: Scale numerical features (methods: standard, minmax, quantile)
- **Discretize**: Bin continuous features (methods: uniform, quantile)
- **ConvertDatetime**: Handle datetime columns (methods: to_timestamp, decompose, ignore)
