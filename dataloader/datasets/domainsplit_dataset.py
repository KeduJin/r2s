import json
import os
import random
from typing import Dict, Union

import lmdb
import polars as pl

try:
    from dataloader.datasets.base_dataset import BaseDataset
    from utils.dataloader_utils import domain_fillin
    from utils.experiments_utils import timer
    from utils.perturb_domain_sequence import perturb_domain, perturb_domain_sequence
except ImportError:
    import sys

    # 使用相对于当前文件的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    sys.path.insert(0, project_root)
    from dataloader.datasets.base_dataset import BaseDataset
    from utils.dataloader_utils import domain_fillin
    from utils.experiments_utils import timer




class DomainSplitDataset(BaseDataset):
    @timer
    def __init__(
        self,
        split_path: str,
        seq_lmdb_path: str,
        domain_lmdb_path: str,
        cluster_lmdb_path: str = None,
        shuffle_domain: bool = False,
        perturb_domain_prob: float = 0.0,
        perturb_domain_max_offset: int = 5,
        need_domain_pieces: bool = False,
        perturb_sequence_prob: float = 0.0,
        seq_mutation_rate: float = 0.05,
        seq_boundary_mutation_rate_multiplier: float = 1.0,
        seq_blosum_temperature: float = 1.0,
        seq_boundary_residue_count: int = 0,
        seq_use_blosum: bool = True,  # 新增：是否使用BLOSUM采样
        logger=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split_path = split_path
        self.seq_lmdb_path = seq_lmdb_path
        self.domain_lmdb_path = domain_lmdb_path
        self.cluster_lmdb_path = cluster_lmdb_path
        self.shuffle_domain = shuffle_domain
        self.perturb_domain_prob = perturb_domain_prob
        self.perturb_domain_max_offset = perturb_domain_max_offset
        self.need_domain_pieces = need_domain_pieces
        # 保存序列扰动参数
        self.perturb_sequence_prob = perturb_sequence_prob
        self.seq_mutation_rate = seq_mutation_rate
        self.seq_boundary_mutation_rate_multiplier = seq_boundary_mutation_rate_multiplier
        self.seq_blosum_temperature = seq_blosum_temperature
        self.seq_boundary_residue_count = seq_boundary_residue_count
        self.seq_use_blosum = seq_use_blosum
        self.seq_env = lmdb.open(
            seq_lmdb_path, lock=False, map_size=1024**4, readonly=True
        )
        self.seq_env_txn = self.seq_env.begin()
        self.domain_env = lmdb.open(
            domain_lmdb_path, lock=False, map_size=1024**4, readonly=True
        )
        self.domain_env_txn = self.domain_env.begin()
        df = pl.read_csv(
            split_path,
            separator="\t",
            has_header=False,
            new_columns=["rep_id", "entry_id"],
        )
        entry_ids = df["entry_id"].to_list()
        rep_ids = df["rep_id"].unique().sort().to_list()

        if cluster_lmdb_path is not None:
            self.cluster_env = lmdb.open(
                cluster_lmdb_path, lock=False, map_size=1024**4, readonly=True
            )
            self.cluster_env_txn = self.cluster_env.begin()
            self.index_mapper = rep_ids
        else:
            self.cluster_env_txn = None
            self.index_mapper = entry_ids

    def __del__(self):
        """Clean up LMDB resources when dataset is destroyed."""
        if hasattr(self, "seq_env") and self.seq_env:
            self.seq_env.close()
        if hasattr(self, "domain_env") and self.domain_env:
            self.domain_env.close()
        if hasattr(self, "cluster_env"):
            self.cluster_env.close()

    def __getitem__(self, idx: int) -> Dict[str, Union[str, list]]:
        if self.cluster_env_txn:
            rep_id = self.index_mapper[idx]
            entry_ids = json.loads(self.cluster_env_txn.get(rep_id.encode()).decode())
            entry_id = random.choice(entry_ids)
        else:
            entry_id = self.index_mapper[idx]

        seq = self.seq_env_txn.get(entry_id.encode()).decode()
        domain_info_list = json.loads(
            self.domain_env_txn.get(entry_id.encode()).decode()
        )

        # 初始化统计信息
        domain_boundary_changed = False
        domain_boundary_change_ratio = 0.0
        domain_mutation_ratio = 0.0

        # check if perturb the domain information
        old_domain_info_list = domain_info_list
        if self.perturb_domain_prob > 0 and random.random() < self.perturb_domain_prob:
            domain_info_list = perturb_domain(
                domain_info_list, seq, self.perturb_domain_max_offset
            )
            # 计算domain边界变化
            domain_boundary_changed = True
            # 计算有多少个氨基酸位置因为边界变化而改变
            # 1. 构建旧的domain覆盖位置集合
            old_positions = set()
            for domain_info in old_domain_info_list:
                for pos_range in domain_info.split("_"):
                    start, end = map(int, pos_range.split("-"))
                    old_positions.update(range(start, end + 1))

            # 2. 构建新的domain覆盖位置集合
            new_positions = set()
            for domain_info in domain_info_list:
                for pos_range in domain_info.split("_"):
                    start, end = map(int, pos_range.split("-"))
                    new_positions.update(range(start, end + 1))

            # 3. 计算有多少个氨基酸位置发生了变化（被添加或被移除）
            changed_positions = old_positions.symmetric_difference(new_positions)
            domain_boundary_change_ratio = len(changed_positions) / len(seq) if len(seq) > 0 else 0.0

        # change domain_info to a patch of sequences (先提取domain)
        domain = []
        original_domain = []  # 保存原始domain用于计算变异率
        for domain_info in domain_info_list:
            domain_seq = domain_fillin(seq, domain_info)
            original_domain.append(domain_seq)
            domain.append(domain_seq)

        # 新增：检查是否对domain序列进行扰动
        domain_mutation_ratio = 0.0
        if self.perturb_sequence_prob > 0 and random.random() < self.perturb_sequence_prob:
            # 对每个domain序列进行扰动
            perturbed_domain = []
            total_mutations = 0
            total_length = 0

            for domain_seq in domain:
                # 对单个domain序列做扰动
                perturbed_seq = perturb_domain_sequence(
                    seq=domain_seq,
                    domain_info_list=["1-" + str(len(domain_seq))],  # 把整个domain当作一个完整序列
                    mutation_rate=self.seq_mutation_rate,
                    boundary_mutation_rate_multiplier=self.seq_boundary_mutation_rate_multiplier,
                    blosum_temperature=self.seq_blosum_temperature,
                    boundary_residue_count=self.seq_boundary_residue_count,
                    use_blosum=self.seq_use_blosum,
                )
                perturbed_domain.append(perturbed_seq)

                # 统计变异数量
                mutations = sum(1 for i in range(len(domain_seq)) if domain_seq[i] != perturbed_seq[i])
                total_mutations += mutations
                total_length += len(domain_seq)

            # 使用扰动后的domain
            domain = perturbed_domain
            # 计算domain的平均变异率
            domain_mutation_ratio = total_mutations / total_length if total_length > 0 else 0.0

        # 添加domain位置信息
        domain_positions = []  # num_domain, domain info (i.e. (start, end))
        for domain_info in domain_info_list:
            positions = []
            for pos_range in domain_info.split("_"):
                start, end = map(int, pos_range.split("-"))
                positions.append((start - 1, end))  # 0-based索引,左闭右开
            domain_positions.extend(positions)

        if self.shuffle_domain:
            # shuffle together the domain, domain_info_list, domain_positions, by the same order
            domain_tuple = list(zip(domain, domain_info_list))
            random.shuffle(domain_tuple)
            domain, domain_info_list = zip(*domain_tuple)

        if self.need_domain_pieces:
            domain_pieces = []
            for single_domain in domain:
                domain_piece = single_domain.split("<unk>")
                domain_pieces.extend(domain_piece)

        return {
            "entry_id": entry_id,
            "seq": seq,
            "domain": domain,
            "domain_pieces": domain_pieces if self.need_domain_pieces else None,
            "domain_info_list": domain_info_list,
            "domain_positions": domain_positions,
            "old_domain_info_list": old_domain_info_list,
            # 新增：扰动统计信息
            "domain_boundary_changed": domain_boundary_changed,
            "domain_boundary_change_ratio": domain_boundary_change_ratio,
            "domain_mutation_ratio": domain_mutation_ratio,
            "original_domain": original_domain
        }