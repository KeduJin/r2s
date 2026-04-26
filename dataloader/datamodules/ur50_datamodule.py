from typing import (
    Optional,
)

from torch.utils.data import Dataset

from dataloader.datamodules.BaseDataModule import BaseDataModule
from dataloader.datasets.ur50_dataset import (
    UniRefDatasetForTesting,
    UniRefHFDataset,
    setup_dataloader,
)

# log = utils.get_logger(__name__)


class UniRefHFDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "data/tape",
        max_tokens: int = 6000,
        max_len: int = 2048,
        num_workers_per_gpu: int = 0,
        num_seqs: int = 40,  # used for testing
        logger=None,
        **kwargs,
    ):
        # super().__init__(**kwargs)
        # this line allows to access init params with 'self.hparams' attribute
        # self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.max_len = max_len
        self.num_workers = num_workers_per_gpu
        self.num_seqs = num_seqs
        self.logger = logger

        self.alphabet = None

        self.train_data: Optional[Dataset] = None
        self.valid_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

        self.setup_flag = False

    # from base data module
    def setup(self, stage: str):
        self.stage = stage
        if not self.setup_flag:
            if stage == "train":
                self.train_dataset = UniRefHFDataset(
                    data_dir=self.data_dir,
                    split="train",
                    max_len=self.max_len,
                    logger=self.logger,
                )
                self.valid_dataset = UniRefHFDataset(
                    data_dir=self.data_dir,
                    split="validation",
                    max_len=self.max_len,
                    logger=self.logger,
                )
            elif stage == "test":
                self.test_dataset = UniRefDatasetForTesting(
                    max_len=self.max_len,
                    num_seqs=self.num_seqs,
                    logger=self.logger,
                )
            self.setup_flag = True

    # def setup(self, stage: Optional[str] = None):
    #     """Load data. Set variables: `self.data_train`, `self.data_val`,
    #     `self.data_test`.

    #     This method is called by lightning when doing `trainer.fit()` and
    #     `trainer.test()`, so be careful not to execute the random split twice!
    #     The `stage` can be used to differentiate whether it's called before
    #     trainer.fit()` or `trainer.test()`.
    #     """

    #     if stage == "fit":
    #         self.train_dataset = UniRefHFDataset(
    #             data_dir=self.data_dir,
    #             split="train",
    #             max_len=self.max_len,
    #         )
    #         self.valid_dataset = UniRefHFDataset(
    #             data_dir=self.data_dir,
    #             split="validation",
    #             max_len=self.max_len,
    #         )
    #     elif stage == "test" or stage == "predict":
    #         self.test_dataset = UniRefDatasetForTesting(
    #             max_len=self.max_len, num_seqs=self.num_seqs
    #         )
    #         self.train_dataset = UniRefDatasetForTesting(
    #             max_len=self.max_len, num_seqs=self.num_seqs
    #         )  # used for deepspeed
    #     else:
    #         raise ValueError(f"Invalid stage: {stage}.")
    #     self.stage = stage

    def train_dataloader(self):
        dl = setup_dataloader(
            self.train_dataset,
            max_tokens=self.max_tokens,
            num_workers=self.num_workers,
            max_len=self.max_len,
            max_batch_size=1
            if self.stage == "test" or self.stage == "predict"
            else 800,
        )
        dl._is_accelerate_prepared = True  # set to True to avoid double wrapping
        return dl

    def val_dataloader(self):
        dl = setup_dataloader(
            self.valid_dataset,
            max_tokens=self.max_tokens,
            num_workers=self.num_workers,
            max_len=self.max_len,
        )
        dl._is_accelerate_prepared = True  # set to True to avoid double wrapping
        return dl

    def test_dataloader(self):
        dl = setup_dataloader(
            self.test_dataset,
            max_tokens=self.max_tokens,
            num_workers=self.num_workers,
            max_len=self.max_len,
            bucket_size=self.num_seqs,
            max_batch_size=self.num_seqs,
        )
        dl._is_accelerate_prepared = True  # set to True to avoid double wrapping
        return dl
