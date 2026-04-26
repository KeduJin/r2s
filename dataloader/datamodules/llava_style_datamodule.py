
try:
    from dataloader.datamodules.BaseDataModule import BaseDataModule
    from dataloader.datasets.llava_style_dataset import LlavaStyleDataset
except ImportError:
    import sys
    sys.path.append("/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain")
    from dataloader.datamodules.BaseDataModule import BaseDataModule
    from dataloader.datasets.llava_style_dataset import LlavaStyleDataset


class LlavaStyleDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        self.train_split_path = kwargs.pop("train_split_path")
        self.val_split_path = kwargs.pop("val_split_path")
        self.test_split_path = kwargs.pop("test_split_path")
        self.cluster_lmdb_path = kwargs.pop("cluster_lmdb_path")
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def set_train_dataset(self):
        self.train_dataset = LlavaStyleDataset(
            split_path=self.train_split_path,
            cluster_lmdb_path=self.cluster_lmdb_path,
            **self.kwargs,
        )

    def set_val_dataset(self):
        self.val_dataset = LlavaStyleDataset(
            split_path=self.val_split_path, cluster_lmdb_path=None, **self.kwargs
        )  # no cluster for validation set

    def set_test_dataset(self):
        self.test_dataset = LlavaStyleDataset(
            split_path=self.test_split_path, cluster_lmdb_path=None, **self.kwargs
        )  # no cluster for test set


if __name__ == "__main__":
    from utils.config_utils import load_config_from_yaml
    from utils.init_utils import construct_class_by_name
    from transformers import EsmTokenizer
    from utils.logger import MyLogger
    from tqdm import tqdm
    config_path = "/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/configs/train/rag-qwen-qwen3_100M-weighting1_1-plddt_filter-dynamicdomainweighting.yaml"
    cfg = load_config_from_yaml(config_path)
    _log = MyLogger()
    tokenizer = EsmTokenizer.from_pretrained("airkingbd/dplm_150m")
    datamodule = construct_class_by_name(**cfg.Datamodule.kwargs.to_dict(), tokenizer=tokenizer, logger=_log)
    datamodule.setup(stage="train")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        target = batch["labels"].clone()
        target = target.masked_fill(target == tokenizer.pad_token_id, -100)
        ###
        unk_mask = (target == tokenizer.unk_token_id)
        if unk_mask.any():
            print(f"WARNING at 1: Found {unk_mask.sum().item()}")
            assert 0