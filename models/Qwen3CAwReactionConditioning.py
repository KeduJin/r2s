import logging
import re
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, EsmTokenizer, GenerationConfig

from utils.metrics import Scalar

from .BaseModel import BaseModel
from .Qwen3.configuration_domainconditioning_qwen3 import Qwen3Config
from .Qwen3.modeling_domainconditioning_qwen3 import Qwen3CAForCausalLM


class Qwen3CAwReactionConditioning(BaseModel):
    def __init__(
        self,
        qwen3_type="Qwen/Qwen3-100M",
        chemberta_model_name_or_path="DeepChem/ChemBERTa-77M-MLM",
        reaction_tokenizer_name_or_path=None,
        seq_tokenizer_name_or_path="airkingbd/dplm_150m",
        freeze_reaction_encoder=False,
        encoder_pack_max_len=1024,
        criterion_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger.info(
            f"{self.__class__.__name__} initialized with qwen3_type: {qwen3_type}, "
            f"chemberta_model_name_or_path: {chemberta_model_name_or_path}"
        )
        self.encoder_pack_max_len = encoder_pack_max_len

        self.tokenizer = EsmTokenizer.from_pretrained(seq_tokenizer_name_or_path)
        reaction_tokenizer_name_or_path = (
            reaction_tokenizer_name_or_path or chemberta_model_name_or_path
        )
        self.condition_tokenizer = AutoTokenizer.from_pretrained(
            reaction_tokenizer_name_or_path
        )
        if self.condition_tokenizer.pad_token is None:
            self.condition_tokenizer.pad_token = (
                self.condition_tokenizer.eos_token
                or self.condition_tokenizer.unk_token
                or self.condition_tokenizer.cls_token
            )

        if qwen3_type == "Qwen/Qwen3-100M":
            config = Qwen3Config(
                architectures=["Qwen3ForCausalLM"],
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=self.tokenizer.cls_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                head_dim=64,
                hidden_act="silu",
                hidden_size=768,
                initializer_range=0.02,
                intermediate_size=2304,
                max_position_embeddings=40960,
                max_window_layers=16,
                model_type="qwen3",
                num_attention_heads=12,
                num_hidden_layers=16,
                num_key_value_heads=6,
                rms_norm_eps=1e-06,
                rope_scaling=None,
                rope_theta=1000000,
                sliding_window=None,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                use_cache=True,
                use_sliding_window=False,
                vocab_size=len(self.tokenizer.get_vocab()),
                add_cross_attention=True,
            )
            self.qwen3 = Qwen3CAForCausalLM(config)
        else:
            config = Qwen3Config.from_pretrained(
                qwen3_type,
                add_cross_attention=True,
                vocab_size=len(self.tokenizer.get_vocab()),
            )
            self.qwen3 = Qwen3CAForCausalLM(config)
        self.qwen3.gradient_checkpointing_enable()

        self.reaction_encoder = AutoModel.from_pretrained(chemberta_model_name_or_path)
        if hasattr(self.reaction_encoder, "gradient_checkpointing_enable"):
            self.reaction_encoder.gradient_checkpointing_enable()
        if (
            hasattr(self.reaction_encoder, "encoder")
            and hasattr(self.reaction_encoder.encoder, "gradient_checkpointing")
        ):
            self.reaction_encoder.encoder.gradient_checkpointing = True

        if freeze_reaction_encoder:
            for param in self.reaction_encoder.parameters():
                param.requires_grad = False

        self.domain_feats_projector = nn.Linear(
            self.reaction_encoder.config.hidden_size, self.qwen3.config.hidden_size
        )
        if criterion_kwargs is not None:
            logging.warning(
                "criterion_kwargs is ignored in Qwen3CAwReactionConditioning. "
                "The model uses the decoder's default causal LM loss."
            )

    def set_objective_and_metrics(self, stage: str = "train", experiment=None):
        if stage == "train":
            train_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "nonpad_ratio": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "nonpad_ratio": Scalar(dist_sync_on_step=True),
            }
            train_metrics = {
                key: value.to(experiment.device) for key, value in train_metrics.items()
            }
            val_metrics = {
                key: value.to(experiment.device) for key, value in val_metrics.items()
            }
            test_metrics = None
        else:
            train_metrics = None
            val_metrics = None
            test_metrics = {
                "gt_seq_softlcs_score": Scalar(dist_sync_on_step=True),
            }
            test_metrics = {
                key: value.to(experiment.device) for key, value in test_metrics.items()
            }
        experiment.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

    def _forward(
        self,
        seq_ids: torch.Tensor,
        reaction_ids: torch.Tensor,
        reaction_masks: torch.Tensor,
        num_reactions_per_protein: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> dict:
        reaction_feats, reaction_feat_masks = self.forward_encoder(
            reaction_ids, reaction_masks, num_reactions_per_protein
        )
        return self.qwen3(
            input_ids=seq_ids,
            encoder_hidden_states=reaction_feats,
            encoder_attention_mask=reaction_feat_masks,
            labels=labels,
            return_dict=True,
        )

    def forward(
        self,
        seq_ids: torch.Tensor,
        reaction_ids: torch.Tensor,
        reaction_masks: torch.Tensor,
        num_reactions_per_protein: torch.Tensor,
        **kwargs,
    ):
        target = seq_ids.clone()
        target = target.masked_fill(target == self.tokenizer.pad_token_id, -100)
        out_dict = self._forward(
            seq_ids=seq_ids,
            reaction_ids=reaction_ids,
            reaction_masks=reaction_masks,
            num_reactions_per_protein=num_reactions_per_protein,
            labels=target,
        )
        sample_size = target.ne(-100).float().sum()
        nonpad_ratio = sample_size / target.numel()
        return {
            "logits": out_dict["logits"],
            "target": target,
            "loss": out_dict["loss"],
            "sample_size": sample_size,
            "nonpad_ratio": nonpad_ratio,
        }

    def forward_encoder(
        self,
        reaction_ids: torch.Tensor,
        reaction_masks: torch.Tensor,
        num_reactions_per_protein: torch.Tensor,
    ):
        encoder_out = self.reaction_encoder(
            input_ids=reaction_ids, attention_mask=reaction_masks
        ).last_hidden_state
        encoder_emb_dim = encoder_out.shape[-1]

        reaction_feats = []
        reaction_feat_masks = []
        start_idx = 0
        for reaction_count in num_reactions_per_protein:
            end_idx = start_idx + reaction_count
            current_encoder_out = encoder_out[start_idx:end_idx]
            current_reaction_masks = reaction_masks[start_idx:end_idx]

            encoder_out_flat = current_encoder_out.reshape(-1, encoder_emb_dim)
            reaction_masks_flat = current_reaction_masks.reshape(-1)

            bool_mask = reaction_masks_flat.bool()
            encoder_out_flat = encoder_out_flat[bool_mask]
            reaction_masks_flat = reaction_masks_flat[bool_mask]

            reaction_feats.append(encoder_out_flat)
            reaction_feat_masks.append(reaction_masks_flat)
            start_idx = end_idx

        reaction_feats_padded = []
        reaction_feat_masks_padded = []
        for reaction_feat, reaction_feat_mask in zip(reaction_feats, reaction_feat_masks):
            if reaction_feat.shape[0] < self.encoder_pack_max_len:
                reaction_feat = F.pad(
                    reaction_feat,
                    (0, 0, 0, self.encoder_pack_max_len - reaction_feat.shape[0]),
                    mode="constant",
                    value=0,
                )
                reaction_feat_mask = F.pad(
                    reaction_feat_mask,
                    (0, self.encoder_pack_max_len - reaction_feat_mask.shape[0]),
                    mode="constant",
                    value=0,
                )
            else:
                reaction_feat = reaction_feat[: self.encoder_pack_max_len, :]
                reaction_feat_mask = reaction_feat_mask[: self.encoder_pack_max_len]

            reaction_feats_padded.append(reaction_feat)
            reaction_feat_masks_padded.append(reaction_feat_mask)

        reaction_feats = torch.stack(reaction_feats_padded)
        reaction_feats = self.domain_feats_projector(reaction_feats)
        reaction_feat_masks = torch.stack(reaction_feat_masks_padded)
        return reaction_feats, reaction_feat_masks

    def initialize_output_tokens(self, bs: int, **kwargs):
        start_id = self.tokenizer.cls_token_id
        input_ids = (
            (torch.zeros((1)) + start_id).unsqueeze(0).repeat(bs, 1)
        )
        input_ids = input_ids.to(torch.long)
        input_ids = input_ids.to(next(self.parameters()).device)
        return input_ids

    def to_list(self, seq: torch.Tensor):
        return [
            seq[i, ...].detach().cpu().numpy().tolist() for i in range(seq.shape[0])
        ]

    def clean_and_format_seq(self, seq: list[str]):
        cleaned_data = []
        for item in seq:
            processed_string = re.sub(r"<cls>", "", item)
            processed_string = re.sub(r"<eos>", "", processed_string)
            processed_string = re.sub(r"<pad>", "", processed_string)
            processed_string = processed_string.replace(" ", "")
            cleaned_data.append(processed_string)
        return cleaned_data

    def generate(
        self,
        reaction_ids: torch.Tensor,
        reaction_masks: torch.Tensor,
        num_reactions_per_protein: torch.Tensor,
        generation_config: GenerationConfig,
        verbose=True,
        **kwargs,
    ):
        encoder_out, encoder_mask = self.forward_encoder(
            reaction_ids, reaction_masks, num_reactions_per_protein
        )
        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        sample_results = self.qwen3.generate(
            generation_config=generation_config,
            num_return_sequences=1,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=encoder_mask,
            return_dict_in_generate=True,
        )
        tokens = self.to_list(sample_results.sequences)
        sequences = self.tokenizer.batch_decode(tokens)
        return {
            "output_seqs": self.clean_and_format_seq(sequences),
        }
