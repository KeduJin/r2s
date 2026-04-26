from typing import Any, Dict

import torch

from .domainsplit_dataset import DomainSplitDataset


class LlavaStyleDataset(DomainSplitDataset):
    """
    This dataset prepares data in a LLaVA-like format for decoder-only models.
    It combines domain information and a protein sequence into a single input string.
    The domain acts as a prompt for the model to generate the sequence.

    Example format: <domain_bos>domain1<domain_eos><domain_bos>domain2<domain_eos>protein_sequence
    """

    def __init__(
        self,
        *args,
        domain_cls_token: str = "<domain_cls>",
        domain_eos_token: str = "<domain_eos>",
        max_prompt_len: int = 1024,
        **kwargs,
    ):
        """
        Args:
            domain_bos_token (str): The token to place at the beginning of the domain prompt.
            domain_eos_token (str): The token to place at the end of the domain prompt.
            max_prompt_len (int): Maximum length for the domain prompt (default: 1024).
            *args, **kwargs: Arguments passed to the parent DomainSplitDataset.
        """
        super().__init__(*args, **kwargs)
        self.domain_cls_token = domain_cls_token
        self.domain_eos_token = domain_eos_token
        self.max_prompt_len = max_prompt_len

        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.eos_token

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Overrides the parent method to return a single combined text string.
        """
        # Retrieve original data from the parent class
        data = super().__getitem__(idx)
        seq = data["seq"]
        domains = data["domain"]  # This is a list of domain strings

        # Format each domain with BOS/EOS tokens and concatenate them
        domain_prompt_parts = []
        for domain in domains:
            domain_prompt_parts.append(
                f"{self.domain_cls_token}{domain}{self.domain_eos_token}"
            )
        domain_prompt = "".join(domain_prompt_parts)

        # Combine into the final text format
        combined_text = f"{domain_prompt}{self.cls_token}{seq}{self.eos_token}"

        return {"text": combined_text, **data}

    def collate(self, batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function to tokenize the combined text and prepare labels
        for a decoder-only language model.
        """

        keys = [key for key in batch[0].keys()]
        dict_batch = {}
        # Here we modify structure token by plddt mask
        for k in keys:
            dict_batch[k] = [dic[k] if k in dic else None for dic in batch]

        # Tokenize the batch of combined texts
        encodings = self.tokenizer(
            dict_batch["text"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            # max_length = max_prompt_len + max_aa_seq_len to accommodate both prompt and sequence
            max_length=self.max_prompt_len + self.max_aa_seq_len,
            add_special_tokens=False,
        )

        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        # Create labels by cloning input_ids
        labels = input_ids.clone()

        # Find the token ID for <cls>
        cls_token_id = self.tokenizer.cls_token_id

        # Mask out the prompt part in the labels
        for i in range(input_ids.shape[0]):
            # Find the position of <cls> token.
            # The prompt consists of all tokens up to and including <cls>.
            cls_indices = (input_ids[i] == cls_token_id).nonzero(as_tuple=True)[0]

            if len(cls_indices) > 0:
                prompt_end_idx = cls_indices[0]  # Use the first occurrence of <cls>
                # Mask tokens from the beginning up to and including <cls>
                labels[i, : prompt_end_idx + 1] = -100
            else:
                # If <cls> token is not found (very rare), ignore this sample.
                labels[i, :] = -100

        # Additionally, mask out padding tokens in the labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            **dict_batch,
        }

        # Call parent's collate to handle domain pieces encoding
        # This will add domain_pieces_ids, domain_pieces_masks, and num_domain_pieces_per_protein
        parent_result = super().collate(batch)

        # Merge parent's domain pieces processing
        if "domain_pieces_ids" in parent_result:
            result["domain_pieces_ids"] = parent_result["domain_pieces_ids"]
            result["domain_pieces_masks"] = parent_result["domain_pieces_masks"]
            result["num_domain_pieces_per_protein"] = parent_result["num_domain_pieces_per_protein"]

        return result
