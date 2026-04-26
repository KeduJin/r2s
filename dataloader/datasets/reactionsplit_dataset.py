import json
import os
import random
from typing import Dict

import lmdb
import polars as pl

try:
    from dataloader.datasets.base_dataset import BaseDataset
    from utils.experiments_utils import timer
except ImportError:
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    sys.path.insert(0, project_root)
    from dataloader.datasets.base_dataset import BaseDataset
    from utils.experiments_utils import timer


class ReactionSplitDataset(BaseDataset):
    _shared_envs = {}
    _shared_env_refcounts = {}

    @classmethod
    def _open_shared_env(cls, path: str):
        normalized_path = os.path.abspath(path)
        env = cls._shared_envs.get(normalized_path)
        if env is None:
            env = lmdb.open(
                normalized_path, lock=False, map_size=1024**4, readonly=True
            )
            cls._shared_envs[normalized_path] = env
            cls._shared_env_refcounts[normalized_path] = 0
        cls._shared_env_refcounts[normalized_path] += 1
        return normalized_path, env

    @classmethod
    def _close_shared_env(cls, path: str):
        normalized_path = os.path.abspath(path)
        env = cls._shared_envs.get(normalized_path)
        if env is None:
            return
        remaining = cls._shared_env_refcounts.get(normalized_path, 0) - 1
        if remaining <= 0:
            env.close()
            cls._shared_envs.pop(normalized_path, None)
            cls._shared_env_refcounts.pop(normalized_path, None)
        else:
            cls._shared_env_refcounts[normalized_path] = remaining

    @timer
    def __init__(
        self,
        split_path: str,
        seq_lmdb_path: str,
        reaction_lmdb_path: str,
        raw_reaction_lmdb_path: str = None,
        source_lmdb_path: str = None,
        cluster_lmdb_path: str = None,
        logger=None,
        **kwargs,
    ):
        kwargs.setdefault("condition_key", "reaction")
        kwargs.setdefault("condition_ids_key", "reaction_ids")
        kwargs.setdefault("condition_masks_key", "reaction_masks")
        kwargs.setdefault("num_conditions_key", "num_reactions_per_protein")
        super().__init__(**kwargs)
        self.split_path = split_path
        self.seq_lmdb_path = seq_lmdb_path
        self.reaction_lmdb_path = reaction_lmdb_path
        self.raw_reaction_lmdb_path = raw_reaction_lmdb_path
        self.source_lmdb_path = source_lmdb_path
        self.cluster_lmdb_path = cluster_lmdb_path
        self.logger = logger

        self.seq_env_path, self.seq_env = self._open_shared_env(seq_lmdb_path)
        self.seq_env_txn = self.seq_env.begin()
        self.reaction_env_path, self.reaction_env = self._open_shared_env(
            reaction_lmdb_path
        )
        self.reaction_env_txn = self.reaction_env.begin()
        self.raw_reaction_env = None
        self.raw_reaction_env_txn = None
        self.raw_reaction_env_path = None
        if raw_reaction_lmdb_path is not None:
            self.raw_reaction_env_path, self.raw_reaction_env = self._open_shared_env(
                raw_reaction_lmdb_path
            )
            self.raw_reaction_env_txn = self.raw_reaction_env.begin()
        self.source_env = None
        self.source_env_txn = None
        self.source_env_path = None
        if source_lmdb_path is not None:
            self.source_env_path, self.source_env = self._open_shared_env(
                source_lmdb_path
            )
            self.source_env_txn = self.source_env.begin()

        df = pl.read_csv(
            split_path,
            separator="\t",
            has_header=False,
            new_columns=["rep_id", "entry_id"],
            schema_overrides={"rep_id": pl.Utf8, "entry_id": pl.Utf8},
        )
        entry_ids = [str(entry_id) for entry_id in df["entry_id"].to_list()]
        rep_ids = [str(rep_id) for rep_id in df["rep_id"].unique().sort().to_list()]
        if cluster_lmdb_path is not None:
            self.cluster_env_path, self.cluster_env = self._open_shared_env(
                cluster_lmdb_path
            )
            self.cluster_env_txn = self.cluster_env.begin()
            self.index_mapper = rep_ids
        else:
            self.cluster_env = None
            self.cluster_env_txn = None
            self.cluster_env_path = None
            self.index_mapper = entry_ids

    def __del__(self):
        if hasattr(self, "seq_env") and self.seq_env:
            self._close_shared_env(self.seq_env_path)
        if hasattr(self, "reaction_env") and self.reaction_env:
            self._close_shared_env(self.reaction_env_path)
        if hasattr(self, "raw_reaction_env") and self.raw_reaction_env:
            self._close_shared_env(self.raw_reaction_env_path)
        if hasattr(self, "source_env") and self.source_env:
            self._close_shared_env(self.source_env_path)
        if hasattr(self, "cluster_env") and self.cluster_env:
            self._close_shared_env(self.cluster_env_path)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        if self.cluster_env_txn is not None:
            rep_id = str(self.index_mapper[idx])
            entry_ids = json.loads(self.cluster_env_txn.get(rep_id.encode()).decode())
            entry_id = str(random.choice(entry_ids))
        else:
            entry_id = str(self.index_mapper[idx])

        entry_key = entry_id.encode()
        seq = self.seq_env_txn.get(entry_key).decode()
        reaction = self.reaction_env_txn.get(entry_key).decode()
        raw_reaction = (
            self.raw_reaction_env_txn.get(entry_key).decode()
            if self.raw_reaction_env_txn is not None
            else reaction
        )
        source = (
            self.source_env_txn.get(entry_key).decode()
            if self.source_env_txn is not None
            else None
        )
        return {
            "entry_id": entry_id,
            "seq": seq,
            "reaction": reaction,
            "raw_reaction": raw_reaction,
            "source": source,
        }
