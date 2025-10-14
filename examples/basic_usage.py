"""
Basic Usage Examples for tabkit

This script demonstrates common usage patterns for the tabkit library.
"""

import numpy as np
import pandas as pd

from tabkit import TableProcessor


def example_1_basic_usage():
    """Example 1: Basic usage with a CSV file"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create some sample data
    data = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 10,
            "income": np.random.randint(30000, 120000, 100),
            "education": np.random.choice(["HS", "BS", "MS", "PhD"], 100),
            "city": np.random.choice(["NYC", "LA", "Chicago", "Houston"], 100),
            "purchased": np.random.choice([0, 1], 100),
        }
    )

    # Save to CSV (in real use, you'd already have this file)
    data.to_csv("/tmp/example_data.csv", index=False)

    # Define configs as plain dicts
    dataset_config = {
        "dataset_name": "customer_data",
        "data_source": "disk",
        "file_path": "/tmp/example_data.csv",
        "file_type": "csv",
        "label_col": "purchased",
    }

    processor_config = {
        "task_kind": "classification",
        "n_splits": 5,
        "random_state": 42,
    }

    # Create and prepare processor
    processor = TableProcessor(dataset_config=dataset_config, config=processor_config)

    processor.prepare()

    # Get data splits
    X_train, y_train = processor.get_split("train")
    X_val, y_val = processor.get_split("val")
    X_test, y_test = processor.get_split("test")

    print(f"Train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"Test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"\nColumn info:")
    for col_info in processor.columns_info:
        print(f"  - {col_info.name}: {col_info.kind} ({col_info.dtype})")


def example_2_custom_pipeline():
    """Example 2: Custom preprocessing pipeline"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Preprocessing Pipeline")
    print("=" * 60)

    # Create sample data with missing values
    data = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 10,
            "feature2": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C", None], 100),
            "target": np.random.randn(100),  # Regression target
        }
    )

    data.to_csv("/tmp/regression_data.csv", index=False)

    dataset_config = {
        "dataset_name": "regression_example",
        "data_source": "disk",
        "file_path": "/tmp/regression_data.csv",
        "file_type": "csv",
        "label_col": "target",
    }

    # Custom preprocessing pipeline
    custom_pipeline = [
        {"class": "Impute", "params": {"method": "median"}},
        {"class": "Encode", "params": {"method": "most_frequent"}},
        {"class": "Scale", "params": {"method": "standard"}},  # Standardize features
    ]

    processor_config = {
        "task_kind": "regression",
        "pipeline": custom_pipeline,
        "n_splits": 3,
        "random_state": 123,
    }

    processor = TableProcessor(dataset_config=dataset_config, config=processor_config)

    processor.prepare()

    X_train, y_train = processor.get_split("train")

    print(f"Train data shape: {X_train.shape}")
    print(f"Features are now scaled:")
    print(X_train.describe())


def example_3_advanced_config():
    """Example 3: Advanced configuration options"""
    print("\n" + "=" * 60)
    print("Example 3: Advanced Configuration")
    print("=" * 60)

    # Create data with multiple feature types
    data = pd.DataFrame(
        {
            "numerical_1": np.random.randn(200),
            "numerical_2": np.random.randn(200),
            "categorical_1": np.random.choice(["A", "B", "C", "D"], 200),
            "categorical_2": np.random.choice(["X", "Y", "Z"], 200),
            "exclude_me": np.random.randn(200),  # We'll exclude this
            "date_feature": pd.date_range("2020-01-01", periods=200),
            "target": np.random.choice([0, 1, 2], 200),  # Multi-class
        }
    )

    data.to_csv("/tmp/multiclass_data.csv", index=False)

    dataset_config = {
        "dataset_name": "multiclass_example",
        "data_source": "disk",
        "file_path": "/tmp/multiclass_data.csv",
        "file_type": "csv",
        "label_col": "target",
    }

    processor_config = {
        "task_kind": "classification",
        "n_splits": 5,
        "split_idx": 0,  # Use first fold
        "n_val_splits": 4,
        "random_state": 42,
        "exclude_columns": ["exclude_me"],  # Exclude specific columns
        "pipeline": [
            {"class": "Impute", "params": {"method": "most_frequent"}},
            {"class": "ConvertDatetime", "params": {"method": "to_timestamp"}},
            {"class": "Encode", "params": {"method": "most_frequent"}},
        ],
    }

    processor = TableProcessor(dataset_config=dataset_config, config=processor_config)

    processor.prepare()

    print(f"Excluded columns: {processor_config['exclude_columns']}")
    print(f"Remaining columns: {processor.col_names}")
    print(f"Number of classes: {processor.label_info.mapping}")

    # Access cached files
    print(f"\nCached data location: {processor.save_dir}")
    print(f"Cache exists: {processor.is_cached}")


def example_4_minimal_config():
    """Example 4: Minimal config using all defaults"""
    print("\n" + "=" * 60)
    print("Example 4: Minimal Configuration (Using Defaults)")
    print("=" * 60)

    # Create simple data
    data = pd.DataFrame(
        {
            "x1": np.random.randn(50),
            "x2": np.random.choice(["A", "B"], 50),
            "y": np.random.choice([0, 1], 50),
        }
    )

    data.to_csv("/tmp/minimal_data.csv", index=False)

    # Minimal config - only required fields
    dataset_config = {
        "dataset_name": "minimal_example",
        "data_source": "disk",
        "file_path": "/tmp/minimal_data.csv",
        "file_type": "csv",
        "label_col": "y",
    }

    # Use all default processor settings
    processor = TableProcessor(
        dataset_config=dataset_config,
        config=None,  # Uses all defaults
    )

    processor.prepare()

    X_train, y_train = processor.get_split("train")
    print(f"Using defaults, got train shape: {X_train.shape}")
    print(f"Default task_kind: classification")
    print(f"Default n_splits: 10")
    print(f"Default pipeline: Impute -> Encode -> ConvertDatetime")


def example_5_accessing_raw_data():
    """Example 5: Accessing raw and processed data"""
    print("\n" + "=" * 60)
    print("Example 5: Accessing Different Data Formats")
    print("=" * 60)

    data = pd.DataFrame(
        {"feature": np.random.randn(30), "target": np.random.choice([0, 1], 30)}
    )

    data.to_csv("/tmp/access_data.csv", index=False)

    dataset_config = {
        "dataset_name": "access_example",
        "data_source": "disk",
        "file_path": "/tmp/access_data.csv",
        "file_type": "csv",
        "label_col": "target",
    }

    processor = TableProcessor(dataset_config=dataset_config)
    processor.prepare()

    # Get raw dataframe (before preprocessing)
    raw_df = processor.get("raw_df")
    print(f"Raw dataframe shape: {raw_df.shape}")

    # Get individual splits
    X_train, y_train = processor.get_split("train")
    X_val, y_val = processor.get_split("val")
    X_test, y_test = processor.get_split("test")

    # Get all data together
    X_all, y_all = processor.get_split("all")
    print(f"All data shape: {X_all.shape}")

    # Access indices
    train_idxs = processor.get("train_idxs")
    val_idxs = processor.get("val_idxs")
    test_idxs = processor.get("test_idxs")
    print(f"Train indices: {len(train_idxs)}")
    print(f"Val indices: {len(val_idxs)}")
    print(f"Test indices: {len(test_idxs)}")

    # Access the fitted pipeline
    pipeline = processor.get("pipeline")
    print(f"Pipeline has {len(pipeline)} transforms")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TABKIT USAGE EXAMPLES")
    print("=" * 60)

    # Run all examples
    example_1_basic_usage()
    example_2_custom_pipeline()
    example_3_advanced_config()
    example_4_minimal_config()
    example_5_accessing_raw_data()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
