import numpy as np
import pandas as pd
import pytest

from tabkit.data.table_processor import TableProcessor
from tabkit.data.transforms import BaseTransform


# A simple transform for testing mock calls
class MockTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        self.fit_transform_called = False
        self.transform_called = False

    def fit(self, X, **kwargs):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.transform_called = True
        return X

    def fit_transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.fit_transform_called = True
        return self.transform(X)


@pytest.fixture
def sample_data():
    X = pd.DataFrame({"numeric": np.random.randn(100), "categorical": ["A", "B"] * 50})
    y = pd.Series([0, 1] * 50, name="target")
    return X, y


@pytest.fixture
def dataset_config():
    # Using a dummy config that doesn't rely on real data sources
    return {
        "dataset_name": "test_dataset",
        "data_source": "disk",
        "file_path": "dummy.csv",
        "file_type": "csv",
    }


@pytest.fixture
def table_processor_config():
    return {
        "pipeline": [
            {"class": "Impute", "params": {"method": "mean"}},
        ],
        "n_splits": 5,
        "n_val_splits": 4,
    }


@pytest.fixture
def processor(dataset_config, table_processor_config, tmp_path):
    proc = TableProcessor(dataset_config, table_processor_config)
    # Point the save directory to a temporary folder for tests
    proc.save_dir = tmp_path / "processed_data"
    return proc


def test_prepare_pipeline_execution(processor, sample_data, mocker):
    """Test that fit_transform is called on train and transform on val/test."""
    X, y = sample_data
    mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

    # Replace real transform with a mock to spy on it
    mock_transform_instance = MockTransform()

    def mock_instantiate_pipeline(config_list):
        if config_list == processor.config["pipeline"]:
            return [mock_transform_instance]
        return []

    mocker.patch.object(
        processor, "_instantiate_pipeline", side_effect=mock_instantiate_pipeline
    )
    processor.prepare()

    # fit_transform should be called once (on train), transform twice (val, test)
    assert mock_transform_instance.fit_transform_called
    assert mock_transform_instance.transform_called


def test_splitting(processor, sample_data, mocker):
    X, y = sample_data
    assert len(X) == 100
    mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

    processor.prepare()

    train_idxs = np.load(processor.save_dir / "train_idxs.npy")
    val_idxs = np.load(processor.save_dir / "val_idxs.npy")
    test_idxs = np.load(processor.save_dir / "test_idxs.npy")

    # With 100 samples, 5 splits -> test is 20.
    # Train is 80. 4 val splits on 80 -> val is 20, train is 60.
    assert len(test_idxs) == 20
    assert len(val_idxs) == 20
    assert len(train_idxs) == 60
    assert len(np.intersect1d(train_idxs, test_idxs)) == 0
    assert len(np.intersect1d(val_idxs, test_idxs)) == 0
    assert len(np.intersect1d(train_idxs, val_idxs)) == 0


def test_stratified_split_fallback(processor, mocker):
    """Test that we fall back to KFold when stratification isn't possible."""
    # Only 1 sample per class, so StratifiedKFold would fail
    X = pd.DataFrame({"feat": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 2, 3], name="target")
    mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

    # Mocks for splitting functions
    def kfold_side_effect(*args, **kwargs):
        # This function will be the side effect for the split method
        # It returns different splits based on the length of the data it receives
        if len(args[0]) == 4:  # First call on the main dataset X
            return iter([(np.array([2, 3]), np.array([0, 1]))])
        elif len(args[0]) == 2:  # Second call on the training indices tr_idxs
            return iter([(np.array([1]), np.array([0]))])
        return iter([])  # Default empty iterator

    mock_kfold_instance = mocker.MagicMock()
    mock_kfold_instance.split.side_effect = kfold_side_effect
    mock_kfold = mocker.patch(
        "tabkit.data.table_processor.KFold", return_value=mock_kfold_instance
    )

    # StratifiedKFold will be called, but we want to test the KFold fallback path
    mock_stratified_kfold_instance = mocker.MagicMock()
    mock_stratified_kfold_instance.split.side_effect = ValueError("Cannot stratify")
    mock_stratified_kfold = mocker.patch(
        "tabkit.data.table_processor.StratifiedKFold",
        return_value=mock_stratified_kfold_instance,
    )

    processor.config["n_splits"] = 2
    processor.config["n_val_splits"] = 2
    processor.prepare()

    # Check that KFold was used as a fallback, and StratifiedKFold was not.
    mock_kfold.assert_called()
    # The if-condition in the code prevents StratifiedKFold from being instantiated.
    mock_stratified_kfold.assert_not_called()


def test_classification_with_float_label_discretizes(processor, mocker):
    """Tests that a continuous label is discretized for classification tasks by default."""
    X = pd.DataFrame({"feat": np.arange(100)})
    # Create a continuous, float-based label
    y = pd.Series(np.linspace(0, 1, 100), name="target")
    mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

    # Configure for classification but provide no explicit label pipeline
    processor.config["task_kind"] = "classification"
    processor.prepare()

    _, y_processed = processor.get_split("train")

    # Check that the output label is now discrete integers
    assert pd.api.types.is_integer_dtype(y_processed.dtype)
    # Check that it has been binned (default n_bins is 4 for Discretize)
    assert y_processed.nunique() <= 4


def test_caching(processor, sample_data, mocker):
    X, y = sample_data
    mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

    # First call, should run preparation and save to cache
    processor.prepare()
    assert processor.is_cached

    # Mock the processing method to ensure it's NOT called on the second run
    mock_get_splits = mocker.patch.object(processor, "_get_splits")

    # Create a new processor instance pointing to the same directory
    new_processor = TableProcessor(processor.dataset_config, processor.config)
    new_processor.save_dir = processor.save_dir

    # This should load from cache and not call the processing method
    new_processor.prepare()

    mock_get_splits.assert_not_called()


def test_ratio_based_splitting(dataset_config, sample_data, mocker, tmp_path):
    """Test ratio-based splitting mode."""
    X, y = sample_data

    # Configure with ratios instead of K-fold
    ratio_config = {
        "test_ratio": 0.2,   # 20% test
        "val_ratio": 0.1,    # 10% val
        "random_state": 42
    }

    processor = TableProcessor(dataset_config, ratio_config)
    processor.save_dir = tmp_path / "ratio_split_data"
    mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

    processor.prepare()

    train_idxs = np.load(processor.save_dir / "train_idxs.npy")
    val_idxs = np.load(processor.save_dir / "val_idxs.npy")
    test_idxs = np.load(processor.save_dir / "test_idxs.npy")

    # Check approximate sizes (allowing for rounding)
    total = len(X)
    assert len(test_idxs) == pytest.approx(total * 0.2, abs=2)
    assert len(val_idxs) == pytest.approx(total * 0.1, abs=2)
    assert len(train_idxs) == pytest.approx(total * 0.7, abs=2)

    # Check no overlap
    assert len(np.intersect1d(train_idxs, test_idxs)) == 0
    assert len(np.intersect1d(val_idxs, test_idxs)) == 0
    assert len(np.intersect1d(train_idxs, val_idxs)) == 0

    # Check all indices are covered
    all_idxs = np.concatenate([train_idxs, val_idxs, test_idxs])
    assert len(np.unique(all_idxs)) == total


def test_ratio_mode_takes_precedence(dataset_config, sample_data, mocker, tmp_path):
    """Test that ratio mode takes precedence when both modes are configured."""
    X, y = sample_data

    # Configure with BOTH ratios and k-fold params
    config = {
        "test_ratio": 0.3,   # Ratio mode
        "val_ratio": 0.2,    # Ratio mode
        "n_splits": 5,       # K-fold mode (should be ignored)
        "split_idx": 0,      # K-fold mode (should be ignored)
        "random_state": 42
    }

    processor = TableProcessor(dataset_config, config)
    processor.save_dir = tmp_path / "precedence_test"
    mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

    processor.prepare()

    test_idxs = np.load(processor.save_dir / "test_idxs.npy")

    # If ratio mode is used, test should be ~30%
    # If K-fold is used, test would be 20% (1/5)
    total = len(X)
    assert len(test_idxs) == pytest.approx(total * 0.3, abs=2)


def test_invalid_ratios_raise_error(dataset_config, sample_data, mocker, tmp_path):
    """Test that invalid ratio combinations raise errors."""
    X, y = sample_data

    # Ratios that sum to >= 1.0 should fail
    invalid_config = {
        "test_ratio": 0.6,
        "val_ratio": 0.5,  # 0.6 + 0.5 = 1.1 > 1.0
        "random_state": 42
    }

    processor = TableProcessor(dataset_config, invalid_config)
    processor.save_dir = tmp_path / "invalid_ratios"
    mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

    with pytest.raises(ValueError, match="must be < 1.0"):
        processor.prepare()
