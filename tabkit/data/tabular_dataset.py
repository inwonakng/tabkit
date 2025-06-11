import torch
from torch.utils.data import Dataset

from .table_processor import TableProcessor


# TODO: need to fix if we ever use this again
class TabularDataset(Dataset):
    feature_idxs: torch.Tensor
    feature_values: torch.Tensor

    def __init__(
        self,
        processor: TableProcessor,
    ):
        self.feature_idxs = feature_idxs
        self.feature_values = feature_values
        self.labels = labels

    def __len__(self):
        return self.feature_idxs.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | torch.Tensor]:
        return dict(
            feature_idxs=self.feature_idxs[idx],
            feature_vals=self.feature_values[idx],
            labels=self.labels[idx],
        )

    def get_collator(self):

        def collate_fn(data: list[dict[str, Any]]):
            return dict(
                feature_idxs=torch.stack([row["feature_idxs"] for row in data]),
                feature_vals=torch.stack([row["feature_vals"] for row in data]),
                labels=torch.stack([row["labels"] for row in data]),
            )

        return collate_fn
