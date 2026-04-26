from dataloader.datamodules.BaseDataModule import BaseDataModule
from dataloader.datasets.domainsplit_dataset import DomainSplitDataset


class DomainSplitDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        self.train_split_path = kwargs.pop("train_split_path")
        self.val_split_path = kwargs.pop("val_split_path")
        self.test_split_path = kwargs.pop("test_split_path")
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def set_train_dataset(self):
        self.train_dataset = DomainSplitDataset(
            split_path=self.train_split_path, **self.kwargs
        )

    def set_val_dataset(self):
        val_kwargs = self.kwargs.copy()
        val_kwargs["perturb_domain_prob"] = 0.0
        val_kwargs["cluster_lmdb_path"] = None
        val_kwargs["shuffle_domain"] = False
        val_kwargs["perturb_sequence_prob"] = 0
        val_kwargs["seq_mutation_rate"] = 0
        val_kwargs["seq_boundary_mutation_rate_multiplier"] = 1
        val_kwargs["seq_blosum_temperature"] = 1
        val_kwargs["seq_boundary_residue_count"] = 0
        val_kwargs["seq_use_blosum"] = False
        self.val_dataset = DomainSplitDataset(
            split_path=self.val_split_path, **val_kwargs
        )  # no cluster for validation set

    def set_test_dataset(self):
        test_kwargs = self.kwargs.copy()
        # no perturbation for test set
        test_kwargs["perturb_domain_prob"] = 0.0
        test_kwargs["cluster_lmdb_path"] = None
        test_kwargs["shuffle_domain"] = False
        test_kwargs["perturb_sequence_prob"] = 0
        test_kwargs["seq_mutation_rate"] = 0
        test_kwargs["seq_boundary_mutation_rate_multiplier"] = 1
        test_kwargs["seq_blosum_temperature"] = 1
        test_kwargs["seq_boundary_residue_count"] = 0
        test_kwargs["seq_use_blosum"] = False
        self.test_dataset = DomainSplitDataset(
            split_path=self.test_split_path, **test_kwargs
        )  # no cluster for test set
