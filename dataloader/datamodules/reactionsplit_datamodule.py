from dataloader.datamodules.BaseDataModule import BaseDataModule
from dataloader.datasets.reactionsplit_dataset import ReactionSplitDataset


class ReactionSplitDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        self.train_split_path = kwargs.pop("train_split_path")
        self.val_split_path = kwargs.pop("val_split_path")
        self.test_split_path = kwargs.pop("test_split_path")
        self.cluster_lmdb_path = kwargs.get("cluster_lmdb_path")
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def set_train_dataset(self):
        self.train_dataset = ReactionSplitDataset(
            split_path=self.train_split_path, **self.kwargs
        )

    def set_val_dataset(self):
        val_kwargs = self.kwargs.copy()
        val_kwargs["cluster_lmdb_path"] = None
        self.val_dataset = ReactionSplitDataset(
            split_path=self.val_split_path, **val_kwargs
        )

    def set_test_dataset(self):
        test_kwargs = self.kwargs.copy()
        test_kwargs["cluster_lmdb_path"] = None
        self.test_dataset = ReactionSplitDataset(
            split_path=self.test_split_path, **test_kwargs
        )
