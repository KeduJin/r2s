from torch.utils.data import DataLoader


class BaseDataModule:
    def __init__(
        self,
        num_workers_per_gpu: int,
        per_gpu_batchsize: int,
        eval_batch_size: int,
        logger=None,
        **kwargs,
    ):
        super().__init__()

        self.num_workers_per_gpu = num_workers_per_gpu
        self.batch_size = per_gpu_batchsize
        self.eval_batch_size = eval_batch_size
        self.setup_flag = False
        self.logger = logger

    # @property
    # def dataset_cls(self):
    #     raise NotImplementedError("return tuple of dataset class")

    # @property
    # def dataset_name(self):
    #     raise NotImplementedError("return name of dataset")

    def setup(self, stage: str):
        if not self.setup_flag:
            if stage == "train":
                self.set_train_dataset()
                self.set_val_dataset()
            elif stage == "test":
                self.set_test_dataset()
            self.logger.info(f"Setup dataset for {stage} stage.")
            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers_per_gpu,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers_per_gpu,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers_per_gpu,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
