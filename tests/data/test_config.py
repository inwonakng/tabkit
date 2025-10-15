"""
Tests for configuration dataclasses and backward compatibility.
"""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from tabkit import DatasetConfig, TableProcessor, TableProcessorConfig


@pytest.fixture
def sample_data():
    X = pd.DataFrame({"numeric": np.random.randn(100), "categorical": ["A", "B"] * 50})
    y = pd.Series([0, 1] * 50, name="target")
    return X, y


class TestDatasetConfig:
    """Test DatasetConfig dataclass."""

    def test_default_values(self):
        """Test that defaults are set correctly."""
        config = DatasetConfig()
        assert config.dataset_name == "default"
        assert config.data_source is None
        assert config.file_path is None
        assert config.label_col is None

    def test_custom_values(self):
        """Test setting custom values."""
        config = DatasetConfig(
            dataset_name="my_dataset",
            data_source="disk",
            file_path="data/train.csv",
            file_type="csv",
            label_col="target",
        )
        assert config.dataset_name == "my_dataset"
        assert config.data_source == "disk"
        assert config.file_path == "data/train.csv"
        assert config.file_type == "csv"
        assert config.label_col == "target"

    def test_asdict_conversion(self):
        """Test conversion to dictionary using asdict."""
        config = DatasetConfig(
            dataset_name="test",
            data_source="openml",
            openml_task_id=123,
        )
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert config_dict["dataset_name"] == "test"
        assert config_dict["data_source"] == "openml"
        assert config_dict["openml_task_id"] == 123

    def test_openml_config(self):
        """Test OpenML-specific configuration."""
        config = DatasetConfig(
            data_source="openml",
            openml_task_id=3917,
            openml_fold_idx=0,
        )
        assert config.data_source == "openml"
        assert config.openml_task_id == 3917
        assert config.openml_fold_idx == 0


class TestTableProcessorConfig:
    """Test TableProcessorConfig dataclass."""

    def test_default_values(self):
        """Test that defaults are set correctly."""
        config = TableProcessorConfig()
        assert config.task_kind == "classification"
        assert config.test_ratio is None
        assert config.val_ratio is None
        assert config.n_splits == 10
        assert config.fold_idx == 0
        assert config.random_state == 0

    def test_ratio_mode_config(self):
        """Test ratio-based splitting configuration."""
        config = TableProcessorConfig(
            test_ratio=0.2,
            val_ratio=0.1,
            random_state=42,
        )
        assert config.test_ratio == 0.2
        assert config.val_ratio == 0.1
        assert config.random_state == 42

    def test_kfold_mode_config(self):
        """Test K-fold splitting configuration."""
        config = TableProcessorConfig(
            n_splits=5,
            fold_idx=2,
            n_val_splits=4,
            val_fold_idx=1,
        )
        assert config.n_splits == 5
        assert config.fold_idx == 2
        assert config.n_val_splits == 4
        assert config.val_fold_idx == 1

    def test_custom_pipeline(self):
        """Test custom pipeline configuration."""
        pipeline = [
            {"class": "Impute", "params": {"method": "mean"}},
            {"class": "Encode", "params": {"method": "label"}},
            {"class": "Scale", "params": {"method": "standard"}},
        ]
        config = TableProcessorConfig(pipeline=pipeline)
        assert config.pipeline == pipeline
        assert len(config.pipeline) == 3

    def test_asdict_conversion(self):
        """Test conversion to dictionary using asdict."""
        config = TableProcessorConfig(
            test_ratio=0.3,
            val_ratio=0.1,
            task_kind="regression",
            random_state=42,
        )
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert config_dict["test_ratio"] == 0.3
        assert config_dict["val_ratio"] == 0.1
        assert config_dict["task_kind"] == "regression"
        assert config_dict["random_state"] == 42


class TestBackwardCompatibility:
    """Test that both dataclass and dict configs work identically."""

    def test_processor_accepts_dataclass(self, sample_data, mocker, tmp_path):
        """Test TableProcessor works with dataclass configs."""
        X, y = sample_data

        dataset_cfg = DatasetConfig(
            dataset_name="test",
            data_source="disk",
            file_path="dummy.csv",
            file_type="csv",
        )
        processor_cfg = TableProcessorConfig(
            test_ratio=0.2,
            val_ratio=0.1,
        )

        processor = TableProcessor(dataset_config=dataset_cfg, config=processor_cfg)
        processor.save_dir = tmp_path / "dataclass_test"
        mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

        # Should prepare without errors
        processor.prepare()
        assert processor.is_cached

    def test_processor_accepts_dict(self, sample_data, mocker, tmp_path):
        """Test TableProcessor works with dict configs."""
        X, y = sample_data

        dataset_cfg = {
            "dataset_name": "test",
            "data_source": "disk",
            "file_path": "dummy.csv",
            "file_type": "csv",
        }
        processor_cfg = {
            "test_ratio": 0.2,
            "val_ratio": 0.1,
        }

        processor = TableProcessor(dataset_config=dataset_cfg, config=processor_cfg)
        processor.save_dir = tmp_path / "dict_test"
        mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

        # Should prepare without errors
        processor.prepare()
        assert processor.is_cached

    def test_dataclass_and_dict_produce_same_result(
        self, sample_data, mocker, tmp_path
    ):
        """Test that dataclass and dict configs produce identical results."""
        X, y = sample_data

        # Processor with dataclass config
        dataclass_cfg = TableProcessorConfig(
            test_ratio=0.2,
            val_ratio=0.1,
            random_state=42,
        )
        processor_dc = TableProcessor(
            dataset_config={"data_source": "disk", "file_path": "dummy.csv"},
            config=dataclass_cfg,
        )
        processor_dc.save_dir = tmp_path / "dataclass"
        mocker.patch.object(processor_dc, "_load_data", return_value=(X, y, None, None))
        processor_dc.prepare()

        # Processor with dict config
        dict_cfg = {
            "test_ratio": 0.2,
            "val_ratio": 0.1,
            "random_state": 42,
        }
        processor_dict = TableProcessor(
            dataset_config={"data_source": "disk", "file_path": "dummy.csv"},
            config=dict_cfg,
        )
        processor_dict.save_dir = tmp_path / "dict"
        mocker.patch.object(
            processor_dict, "_load_data", return_value=(X, y, None, None)
        )
        processor_dict.prepare()

        # Load split indices and compare
        train_dc = np.load(processor_dc.save_dir / "train_idxs.npy")
        train_dict = np.load(processor_dict.save_dir / "train_idxs.npy")
        np.testing.assert_array_equal(train_dc, train_dict)

        test_dc = np.load(processor_dc.save_dir / "test_idxs.npy")
        test_dict = np.load(processor_dict.save_dir / "test_idxs.npy")
        np.testing.assert_array_equal(test_dc, test_dict)

    def test_mixed_dataclass_and_dict(self, sample_data, mocker, tmp_path):
        """Test mixing dataclass and dict configs."""
        X, y = sample_data

        # Dataclass for dataset, dict for processor
        processor1 = TableProcessor(
            dataset_config=DatasetConfig(
                data_source="disk", file_path="dummy.csv", label_col="target"
            ),
            config={"test_ratio": 0.2, "val_ratio": 0.1},
        )
        processor1.save_dir = tmp_path / "mixed1"
        mocker.patch.object(processor1, "_load_data", return_value=(X, y, None, None))
        processor1.prepare()
        assert processor1.is_cached

        # Dict for dataset, dataclass for processor
        processor2 = TableProcessor(
            dataset_config={"data_source": "disk", "file_path": "dummy.csv"},
            config=TableProcessorConfig(test_ratio=0.2, val_ratio=0.1),
        )
        processor2.save_dir = tmp_path / "mixed2"
        mocker.patch.object(processor2, "_load_data", return_value=(X, y, None, None))
        processor2.prepare()
        assert processor2.is_cached

class TestDataclassModification:
    """Test that dataclasses can be modified after creation."""

    def test_modify_config_after_creation(self):
        """Test modifying config fields after instantiation."""
        config = TableProcessorConfig()
        assert config.test_ratio is None

        # Modify fields
        config.test_ratio = 0.3
        config.exclude_columns = ["id", "timestamp"]
        config.sample_n_rows = 5000

        assert config.test_ratio == 0.3
        assert config.exclude_columns == ["id", "timestamp"]
        assert config.sample_n_rows == 5000

    def test_modify_and_use_with_processor(self, sample_data, mocker, tmp_path):
        """Test that modified configs work with TableProcessor."""
        X, y = sample_data

        # Start with config
        config = TableProcessorConfig(n_splits=5)

        # Customize it
        config.random_state = 999
        config.pipeline = [{"class": "Impute", "params": {"method": "mean"}}]

        processor = TableProcessor(
            dataset_config={"data_source": "disk", "file_path": "dummy.csv"},
            config=config,
        )
        processor.save_dir = tmp_path / "modified"
        mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

        # Should work without errors
        processor.prepare()
        assert processor.is_cached
        assert processor.config.random_state == 999


class TestConfigValidation:
    """Test configuration validation and edge cases."""

    def test_none_config_uses_defaults(self, sample_data, mocker, tmp_path):
        """Test that None config uses default values."""
        X, y = sample_data

        processor = TableProcessor(
            dataset_config={"data_source": "disk", "file_path": "dummy.csv"},
            config=None,  # Explicitly None
        )
        processor.save_dir = tmp_path / "none_config"
        mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

        processor.prepare()
        # Should use default K-fold splitting
        assert processor.config.n_splits == 10

    def test_empty_dict_config_uses_defaults(self, sample_data, mocker, tmp_path):
        """Test that empty dict uses default values."""
        X, y = sample_data

        processor = TableProcessor(
            dataset_config={"data_source": "disk", "file_path": "dummy.csv"},
            config={},  # Empty dict
        )
        processor.save_dir = tmp_path / "empty_config"
        mocker.patch.object(processor, "_load_data", return_value=(X, y, None, None))

        processor.prepare()
        assert processor.config.n_splits == 10
        assert processor.config.task_kind == "classification"
