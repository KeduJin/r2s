import torch
from torch.nn.utils.rnn import pad_sequence

from .base_dataset import BaseDataset


class ReactionDataset(BaseDataset):
    def __init__(
        self,
        split_path: str,
        substrate_embedding_path: str,
        product_embedding_path: str,
        seq_embedding_path: str = None,
        seq_embedding_lookup_max_len: int = 5000,
        max_reaction_atom_tokens: int = None,
        target_seq_type: str = "satoken",
        logger=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split_path = split_path
        self.substrate_embedding_path = substrate_embedding_path
        self.product_embedding_path = product_embedding_path
        self.seq_embedding_path = seq_embedding_path
        self.seq_embedding_lookup_max_len = seq_embedding_lookup_max_len
        self.max_reaction_atom_tokens = max_reaction_atom_tokens
        self.target_seq_type = (target_seq_type or "satoken").lower()
        self.logger = logger

        raw_samples = torch.load(split_path, map_location="cpu")
        self.samples = []
        for entry_id, value in raw_samples.items():
            if not isinstance(value, (tuple, list)) or len(value) < 3:
                raise ValueError(
                    f"Unsupported sample format in {split_path}: entry_id={entry_id}, value={type(value)}"
                )
            substrate, product, seq = value[0], value[1], value[-1]
            if not all(isinstance(v, str) for v in (substrate, product, seq)):
                raise ValueError(
                    f"Unsupported sample values in {split_path}: entry_id={entry_id}, value={value}"
                )
            self.samples.append(
                {
                    "entry_id": str(entry_id),
                    "substrate": substrate,
                    "product": product,
                    "seq": seq,
                }
            )

        self.index_mapper = list(range(len(self.samples)))
        self.substrate_embedding = torch.load(
            substrate_embedding_path, map_location="cpu"
        )
        self.product_embedding = torch.load(product_embedding_path, map_location="cpu")
        self.seq_embedding = (
            torch.load(seq_embedding_path, map_location="cpu")
            if seq_embedding_path not in [None, "None"]
            else None
        )

        if self.logger is not None:
            self.logger.info(
                f"ReactionDataset loaded {len(self.samples)} samples from {split_path}, "
                f"target_seq_type={self.target_seq_type}"
            )

    def _convert_target_seq(self, seq: str) -> str:
        if self.target_seq_type in ["satoken", "full"]:
            return seq
        if self.target_seq_type == "sotoken":
            if len(seq) >= 2:
                structure_only = seq[1::2]
                if len(structure_only) > 0:
                    return structure_only
            return "".join(ch for ch in seq if ch.islower() or ch == "#")
        raise ValueError(
            f"Unsupported target_seq_type={self.target_seq_type}. "
            f"Expected one of ['satoken', 'sotoken', 'full']"
        )

    def _truncate_atom_tokens(self, atom_tokens: torch.Tensor) -> torch.Tensor:
        if (
            self.max_reaction_atom_tokens is None
            or atom_tokens.shape[0] <= self.max_reaction_atom_tokens
        ):
            return atom_tokens

        x = atom_tokens.transpose(0, 1).unsqueeze(0)
        x = torch.nn.functional.adaptive_avg_pool1d(x, self.max_reaction_atom_tokens)
        return x.squeeze(0).transpose(0, 1).contiguous()

    def __getitem__(self, idx):
        sample = self.samples[self.index_mapper[idx]]
        substrate_atom_tokens = self.substrate_embedding[sample["substrate"]].float()
        product_atom_tokens = self.product_embedding[sample["product"]].float()

        substrate_atom_tokens = self._truncate_atom_tokens(substrate_atom_tokens)
        product_atom_tokens = self._truncate_atom_tokens(product_atom_tokens)

        target_seq = self._convert_target_seq(sample["seq"])
        seq_embedding_target = None
        if self.seq_embedding is not None:
            full_seq = target_seq
            truncated_lookup_seq = full_seq[: self.seq_embedding_lookup_max_len]
            if truncated_lookup_seq in self.seq_embedding:
                seq_embedding_target = self.seq_embedding[truncated_lookup_seq].float()
            elif full_seq in self.seq_embedding:
                seq_embedding_target = self.seq_embedding[full_seq].float()
            else:
                raise KeyError(
                    f"Sequence embedding not found for entry_id={sample['entry_id']} "
                    f"in {self.seq_embedding_path}"
                )

        return {
            "entry_id": sample["entry_id"],
            "substrate": sample["substrate"],
            "product": sample["product"],
            "seq": target_seq,
            "substrate_atom_tokens": substrate_atom_tokens,
            "product_atom_tokens": product_atom_tokens,
            "seq_embedding_target": seq_embedding_target,
        }

    def collate(self, batch):
        dict_batch = {
            key: [sample[key] if key in sample else None for sample in batch]
            for key in batch[0].keys()
        }

        if self.tokenizer is not None and "seq" in dict_batch:
            encodings = self.tokenizer(
                dict_batch["seq"],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_aa_seq_len,
                padding="longest",
            )
            if self.target_seq_type == "full":
                unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
                cls_token_id = getattr(self.tokenizer, "cls_token_id", None)
                eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                suspicious_samples = []
                for seq, seq_ids in zip(dict_batch["seq"], encodings.input_ids):
                    raw_seq = seq.replace(" ", "")
                    if len(raw_seq) <= 1:
                        continue
                    content_ids = [
                        token_id
                        for token_id in seq_ids.tolist()
                        if token_id not in [cls_token_id, eos_token_id, self.tokenizer.pad_token_id]
                    ]
                    if content_ids and unk_token_id is not None and all(
                        token_id == unk_token_id for token_id in content_ids
                    ):
                        suspicious_samples.append(raw_seq[:32])
                if suspicious_samples:
                    raise ValueError(
                        "Pure sequence targets were tokenized entirely as <unk>. "
                        f"target_seq_type=full requires a sequence tokenizer (for example "
                        f"/jinkedu/enzyme-design/airkingbd/dplm_150m), got tokenizer={type(self.tokenizer).__name__}. "
                        f"Example sequences: {suspicious_samples[:3]}"
                    )
            dict_batch.update(
                {
                    "seq_ids": encodings.input_ids,
                    "seq_masks": encodings.attention_mask,
                }
            )

        substrate_lengths = torch.tensor(
            [tokens.shape[0] for tokens in dict_batch["substrate_atom_tokens"]],
            dtype=torch.long,
        )
        product_lengths = torch.tensor(
            [tokens.shape[0] for tokens in dict_batch["product_atom_tokens"]],
            dtype=torch.long,
        )

        substrate_atom_tokens = pad_sequence(
            dict_batch["substrate_atom_tokens"], batch_first=True, padding_value=0.0
        )
        product_atom_tokens = pad_sequence(
            dict_batch["product_atom_tokens"], batch_first=True, padding_value=0.0
        )

        substrate_atom_masks = (
            torch.arange(substrate_atom_tokens.shape[1]).unsqueeze(0)
            < substrate_lengths.unsqueeze(1)
        ).long()
        product_atom_masks = (
            torch.arange(product_atom_tokens.shape[1]).unsqueeze(0)
            < product_lengths.unsqueeze(1)
        ).long()

        dict_batch.update(
            {
                "substrate_atom_tokens": substrate_atom_tokens,
                "substrate_atom_masks": substrate_atom_masks,
                "product_atom_tokens": product_atom_tokens,
                "product_atom_masks": product_atom_masks,
            }
        )

        if dict_batch.get("seq_embedding_target", [None])[0] is not None:
            dict_batch["seq_embedding_target"] = torch.stack(
                dict_batch["seq_embedding_target"]
            )

        if "substrate" in dict_batch and "product" in dict_batch:
            dict_batch["reaction"] = [
                f"{s}>>{p}"
                for s, p in zip(dict_batch["substrate"], dict_batch["product"])
            ]
            dict_batch["raw_reaction"] = list(dict_batch["reaction"])

        return dict_batch


class ReactionDatasetView(torch.utils.data.Dataset):
    def __init__(self, dataset: ReactionDataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        self.collate = dataset.collate

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
