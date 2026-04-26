from typing import Any, Dict

import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        max_aa_seq_len=1024,
        condition_tokenizer=None,
        condition_max_len=512,
        condition_key=None,
        condition_ids_key=None,
        condition_masks_key=None,
        num_conditions_key=None,
        **kwargs,
    ):
        super().__init__()
        self.max_aa_seq_len = max_aa_seq_len
        self.tokenizer = tokenizer
        self.condition_tokenizer = condition_tokenizer or tokenizer
        self.condition_max_len = condition_max_len
        self.condition_key = condition_key
        self.condition_ids_key = condition_ids_key
        self.condition_masks_key = condition_masks_key
        self.num_conditions_key = num_conditions_key

    def __len__(self):
        return len(self.index_mapper)

    def _infer_condition_key(self, dict_batch: Dict[str, Any]) -> str | None:
        if self.condition_key is not None:
            return self.condition_key
        if "domain" in dict_batch:
            return "domain"
        if "reaction" in dict_batch:
            return "reaction"
        return None

    def _normalize_condition_batch(self, values: list[Any]) -> list[list[str]]:
        normalized = []
        for value in values:
            if value is None:
                normalized.append([])
            elif isinstance(value, (list, tuple)):
                normalized.append(list(value))
            else:
                normalized.append([value])
        return normalized

    def _default_num_conditions_key(self, condition_key: str) -> str:
        suffix = condition_key if condition_key.endswith("s") else condition_key + "s"
        return f"num_{suffix}_per_protein"

    def collate(self, batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        for domain, we return a list (which length is the batch size), each element is a tensor of shape [num_domain, seq_len]
        """
        keys = [key for key in batch[0].keys()]
        # dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        dict_batch = {}
        # Here we modify structure token by plddt mask
        for k in keys:
            dict_batch[k] = [dic[k] if k in dic else None for dic in batch]

        # encode seq and domain
        if self.tokenizer is not None:
            if "seq" in dict_batch:
                encodings = self.tokenizer(
                    dict_batch["seq"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_aa_seq_len,
                    padding="longest",  # max_length,longest
                )
                dict_batch.update(
                    {
                        "seq_ids": encodings.input_ids,
                        "seq_masks": encodings.attention_mask,
                    }
                )

        condition_key = self._infer_condition_key(dict_batch)
        if condition_key is not None and self.condition_tokenizer is not None:
            condition_batches = self._normalize_condition_batch(dict_batch[condition_key])
            num_conditions_per_protein = torch.tensor(
                [len(conditions) for conditions in condition_batches]
            )

            # calculate num_domain_pieces_per_protein
            if condition_key == "domain" and "domain_positions" in dict_batch:
                num_domain_pieces_per_protein = torch.tensor(
                    [
                        len(domain_position_list)
                        for domain_position_list in dict_batch["domain_positions"]
                    ]
                )
            elif condition_key == "domain":
                num_domain_pieces_per_protein = []
                for domain_list in condition_batches:
                    cur_domain_pieces_num = 0
                    for domain in domain_list:
                        cur_domain_pieces_num += len(domain.split("<unk>"))
                    num_domain_pieces_per_protein.append(cur_domain_pieces_num)
                num_domain_pieces_per_protein = torch.tensor(num_domain_pieces_per_protein)
            else:
                num_domain_pieces_per_protein = None

            all_conditions = []
            for conditions_per_protein in condition_batches:
                all_conditions.extend(conditions_per_protein)

            if all_conditions:
                encodings = self.condition_tokenizer(
                    all_conditions,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.condition_max_len,
                    padding="longest",
                )
                condition_ids = encodings.input_ids
                condition_masks = encodings.attention_mask
            else:
                condition_ids = torch.empty((0, 0), dtype=torch.long)
                condition_masks = torch.empty((0, 0), dtype=torch.long)

            condition_ids_key = self.condition_ids_key or f"{condition_key}_ids"
            condition_masks_key = self.condition_masks_key or f"{condition_key}_masks"
            num_conditions_key = (
                self.num_conditions_key or self._default_num_conditions_key(condition_key)
            )

            dict_batch.update(
                {
                    condition_ids_key: condition_ids,
                    condition_masks_key: condition_masks,
                    num_conditions_key: num_conditions_per_protein,
                }
            )
            if num_domain_pieces_per_protein is not None:
                dict_batch["num_domain_pieces_per_protein"] = num_domain_pieces_per_protein

            # encode domain pieces
            if "domain_pieces" in dict_batch and dict_batch["domain_pieces"][0] is not None:
                all_domain_pieces = []
                for domain_pieces in dict_batch["domain_pieces"]:
                    all_domain_pieces.extend(domain_pieces)
                encodings = self.condition_tokenizer(
                    all_domain_pieces,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.condition_max_len,
                    padding="longest",  # max_length,longest
                )
                domain_pieces_ids = encodings.input_ids
                domain_pieces_masks = encodings.attention_mask
                dict_batch.update(
                    {
                        "domain_pieces_ids": domain_pieces_ids,
                        "domain_pieces_masks": domain_pieces_masks,
                    }
                )

        return dict_batch
