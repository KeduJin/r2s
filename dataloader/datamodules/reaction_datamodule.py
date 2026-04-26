import torch

from dataloader.datamodules.BaseDataModule import BaseDataModule
from dataloader.datasets.reaction_dataset import ReactionDataset, ReactionDatasetView


class ReactionDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        self.train_split_path = kwargs.pop("train_split_path")
        self.val_split_path = kwargs.pop("val_split_path", None)
        self.test_split_path = kwargs.pop("test_split_path", None)
        self.train_ratio = kwargs.pop("train_ratio", None)
        self.val_ratio = kwargs.pop("val_ratio", 0.02)
        self.test_ratio = kwargs.pop("test_ratio", 0.0)
        self.split_seed = kwargs.pop("split_seed", 123)
        self.kwargs = kwargs
        self._shared_train_dataset = None
        self._cached_val_dataset = None
        self._cached_test_dataset = None
        super().__init__(**kwargs)

    def _use_shared_three_way_split(self):
        return self.val_split_path in [None, "None"] and self.test_split_path in [
            None,
            "None",
        ]

    def _build_shared_splits(self):
        if self._shared_train_dataset is not None:
            return

        shared_dataset = ReactionDataset(split_path=self.train_split_path, **self.kwargs)
        generator = torch.Generator().manual_seed(self.split_seed)
        permuted_indices = torch.randperm(
            len(shared_dataset), generator=generator
        ).tolist()

        if self._use_shared_three_way_split():
            val_ratio = float(self.val_ratio or 0.0)
            test_ratio = float(self.test_ratio or 0.0)
            train_ratio = (
                float(self.train_ratio)
                if self.train_ratio is not None
                else 1.0 - val_ratio - test_ratio
            )

            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(
                    f"train/val/test ratios must sum to 1.0, got {total_ratio:.6f}"
                )
            if min(train_ratio, val_ratio, test_ratio) < 0:
                raise ValueError("train/val/test ratios must be non-negative")

            total_size = len(shared_dataset)
            train_size = int(total_size * train_ratio)
            val_size = int(total_size * val_ratio)
            train_end = train_size
            val_end = train_end + val_size
            train_indices = permuted_indices[:train_end]
            val_indices = permuted_indices[train_end:val_end]
            test_indices = permuted_indices[val_end:]

            self._shared_train_dataset = shared_dataset
            self.train_dataset = ReactionDatasetView(shared_dataset, train_indices)
            self._cached_val_dataset = ReactionDatasetView(shared_dataset, val_indices)
            self._cached_test_dataset = ReactionDatasetView(shared_dataset, test_indices)
            return

        if self.val_split_path in [None, "None"]:
            if self.val_ratio is None or self.val_ratio <= 0:
                train_indices = list(range(len(shared_dataset)))
                val_indices = list(range(len(shared_dataset)))
            else:
                val_size = max(1, int(len(shared_dataset) * self.val_ratio))
                train_indices = permuted_indices[val_size:]
                val_indices = permuted_indices[:val_size]

            self._shared_train_dataset = shared_dataset
            self.train_dataset = ReactionDatasetView(shared_dataset, train_indices)
            self._cached_val_dataset = ReactionDatasetView(shared_dataset, val_indices)

    def set_train_dataset(self):
        if self.val_split_path in [None, "None"]:
            self._build_shared_splits()
        else:
            self.train_dataset = ReactionDataset(
                split_path=self.train_split_path, **self.kwargs
            )

    def set_val_dataset(self):
        if self._cached_val_dataset is not None:
            self.val_dataset = self._cached_val_dataset
        else:
            self.val_dataset = ReactionDataset(
                split_path=self.val_split_path, **self.kwargs
            )

    def set_test_dataset(self):
        if self._cached_test_dataset is not None:
            self.test_dataset = self._cached_test_dataset
        elif self._use_shared_three_way_split():
            self._build_shared_splits()
            self.test_dataset = self._cached_test_dataset
        else:
            self.test_dataset = ReactionDataset(
                split_path=self.test_split_path, **self.kwargs
            )
