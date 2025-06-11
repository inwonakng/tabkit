# Table Toolkit (tabkit)

A python library for consistent preprocessing of tabular data. Handles column
type inference, missing value imputation, feature binning, stratified
split/sampling and more in a configuration-driven manner. I made this toolkit because I needed a way to reliably preproecess/cache datasets in a reproducible manner.

## Installation

```
pip install git+https://github.com/inwonakng/tabkit.git@main
```

## Example Usage

```yaml
# config/dataset/DATASET.yaml
dataset_name: example_dataset
data_source: parquet
parquet_path: some_path_to_a_parquet_file.parquet
```

```yaml
# config/preprocess/PREPROCESS.yaml
# note that this is just the defaults.
fold_idx: 0
n_splits: 10
scale_cont: False
cont_scale_method: quantile
cont_scale_max_quantiles: 1000
bin_cont: False
cont_bin_strat: quantile
n_bins: 32
handle_cat_unseen: most_frequent
cat_unseen_fill: Unknown
handle_cat_missing: most_frequent
cat_missing_fill: Missing
handle_cont_missing: median
cont_strat_n_bins: 8
cont_strat_bin_strat: quantile
task_kind: classification
random_state: 0

```

```python
from tabkit import TableProcessor, TableProcessorConfig, DatasetConfig

processor = TableProcessor(
dataset_config = DatasetConfig.from_yaml("config/dataset/DATASET.yaml")
  config = TableProcessorConfig.from_yaml("config/preprocess/PREPROCESS.yaml")
)

# this is the function that does all the heavy lifting. Will cache the results in a hashed directory. So if using in distributed, we only need to process once and read from cache for subsequent runs.
processor.prepare() 

X,y = processor.load_split("train")
...
```
