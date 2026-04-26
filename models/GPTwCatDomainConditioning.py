import re
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    EsmModel,
    EsmTokenizer,
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
)

from models.BaseModel import BaseModel
from utils.init_utils import construct_class_by_name
from utils.metrics import Scalar


class GPTwCatDomainConditioning(BaseModel):
    def __init__(
        self,
        gpt_type="gpt2-amino_acid",
        esm_type="facebook/esm2_t12_35M_UR50D",
        criterion_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger.info(f"GPTwSeqConditioning initialized with gpt_type: {gpt_type}")

        self.tokenizer = EsmTokenizer.from_pretrained("airkingbd/dplm_150m")
        # Since we need to use the cross attention, we need to set the use_cache to False and is_decoder to True
        gpt2_config = GPT2Config.from_pretrained(
            gpt_type,
            add_cross_attention=True,
            cache_dir="~/my_project/checkpoints",
            local_files_only=True,
            vocab_size=len(self.tokenizer.get_vocab()),
        )
        self.gpt2 = GPT2LMHeadModel(gpt2_config)
        # self.gpt2.transformer.gradient_checkpointing = True
        self.gpt2.gradient_checkpointing_enable()

        self.esm_encoder = EsmModel.from_pretrained(esm_type)
        self.esm_encoder.encoder.gradient_checkpointing = True
        self.esm_encoder.gradient_checkpointing_enable()

        self.esm_encoder.pooler.dense.weight.requires_grad = False
        self.esm_encoder.pooler.dense.bias.requires_grad = False
        self.esm_encoder.contact_head.regression.weight.requires_grad = False
        self.esm_encoder.contact_head.regression.bias.requires_grad = False
        self.esm_encoder.embeddings.position_embeddings.weight.requires_grad = False

        self.domain_feats_projector = nn.Linear(
            self.esm_encoder.config.hidden_size, self.gpt2.config.hidden_size
        )
        if criterion_kwargs is not None:
            self.criterion = construct_class_by_name(
                **criterion_kwargs, logger=self.logger
            )
        else:
            self.criterion = None
        # self.cfg = self.dplm.cfg

    def set_objective_and_metrics(self, stage: str = "train", experiment=None):
        # we have loss, nll_loss, ppl, fullseq_loss, fullseq_nll_loss, bsz, sample_size, sample_ratio, nonpad_ratio, weight_diff_loss
        if stage == "train":
            train_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "domain_loss": Scalar(dist_sync_on_step=True),
                "non_domain_loss": Scalar(dist_sync_on_step=True),
                "domain_weight": Scalar(dist_sync_on_step=True),
                "non_domain_weight": Scalar(dist_sync_on_step=True),
                "unweighted_loss": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "domain_loss": Scalar(dist_sync_on_step=True),
                "non_domain_loss": Scalar(dist_sync_on_step=True),
                "domain_weight": Scalar(dist_sync_on_step=True),
                "non_domain_weight": Scalar(dist_sync_on_step=True),
                "unweighted_loss": Scalar(dist_sync_on_step=True),
            }
            train_metrics = {
                k: v.to(experiment.device) for k, v in train_metrics.items()
            }
            val_metrics = {k: v.to(experiment.device) for k, v in val_metrics.items()}
            test_metrics = None
        else:
            train_metrics = None
            val_metrics = None
            test_metrics = {
                "softlcs_score": Scalar(dist_sync_on_step=True),
                "ref_softlcs_score": Scalar(dist_sync_on_step=True),
                "gt_seq_softlcs_score": Scalar(dist_sync_on_step=True),
            }
            test_metrics = {k: v.to(experiment.device) for k, v in test_metrics.items()}
        experiment.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

    def _forward(
        self,
        seq_ids: torch.Tensor,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> dict:
        domain_feats, domain_feat_masks = self.forward_encoder(
            domain_ids, domain_masks, num_domains_per_protein
        )
        out_dict = self.gpt2(
            input_ids=seq_ids,
            encoder_hidden_states=domain_feats,
            encoder_attention_mask=domain_feat_masks,
            labels=labels,
            return_dict=True,
        )
        return out_dict  # [bs, seq_len, vocab_size]

    def forward(
        self,
        seq_ids: torch.Tensor,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
        **kwargs,
    ):
        """
        this function is used for training, computing the loss
        seq_ids: [bs, seq_len]
        kwargs: other arguments

        return:
            logits: [bs, seq_len, vocab_size]
            target: [bs, seq_len]
            loss_mask: [bs, seq_len]
            loss: the loss value
        """
        target = seq_ids.clone()
        target = target.masked_fill(target == self.tokenizer.pad_token_id, -100)

        out_dict = self._forward(
            seq_ids, domain_ids, domain_masks, num_domains_per_protein, labels=target
        )
        logits = out_dict["logits"]
        if self.criterion:
            domain_mask = self._create_domain_mask(seq_ids, kwargs["domain_positions"])
            loss, logging_output = self.criterion(logits, target, domain_mask)
        else:
            loss = out_dict["loss"]
            logging_output = {}
        return {"logits": logits, "target": target, "loss": loss, **logging_output}

    def _create_domain_mask(
        self,
        seq_ids: torch.Tensor,
        domain_positions: list[list[tuple[int, int]]],
    ) -> torch.Tensor:
        """
        根据domain信息创建domain_mask
        Args:
            seq_ids: [bs, seq_len] 序列
            domain_positions: bs, num_domain, domain_positions
        Returns:
            domain_mask: [bs, seq_len] 1表示domain区域，0表示非domain区域
        """
        batch_size, seq_len = seq_ids.shape
        domain_mask = torch.zeros(batch_size, seq_len, device=seq_ids.device)

        for i, positions in enumerate(domain_positions):
            for start, end in positions:
                domain_mask[i, start + 1 : end + 1] = (
                    1  # shift by 1 to exclude the bos token
                )

        return domain_mask

    def forward_encoder(
        self,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
    ):
        # cat domain ids and domain masks
        # Noted that we do not care positional embededings
        start_idx = 0
        rebatched_domain_ids = []
        rebatched_domain_masks = []
        for i, domain_count in enumerate(num_domains_per_protein):
            end_idx = start_idx + domain_count

            # 提取当前data point的encoder输出
            current_domain_ids = domain_ids[start_idx:end_idx]  # [num_domain, seq_len]
            current_domain_masks = domain_masks[
                start_idx:end_idx
            ]  # [num_domain, seq_len]

            # 重塑并去除padding
            domain_ids_flat = current_domain_ids.reshape(-1)  # [num_domain*seq_len]
            domain_masks_flat = current_domain_masks.reshape(-1)  # [num_domain*seq_len]

            # 去除padding
            bool_mask = domain_masks_flat.bool()
            domain_ids_flat = domain_ids_flat[bool_mask]
            domain_masks_flat = domain_masks_flat[bool_mask]

            rebatched_domain_ids.append(domain_ids_flat)
            rebatched_domain_masks.append(domain_masks_flat)

            start_idx = end_idx

        # 第五步：重新padding到1024长度
        domain_ids_padded = []
        domain_masks_padded = []

        for _domain_ids, _domain_masks in zip(
            rebatched_domain_ids, rebatched_domain_masks
        ):
            if _domain_ids.shape[0] < 1024:
                _domain_ids = F.pad(
                    _domain_ids,
                    (0, 1024 - _domain_ids.shape[0]),
                    mode="constant",
                    value=1,
                )  # 1 is the padding token
                _domain_masks = F.pad(
                    _domain_masks,
                    (0, 1024 - _domain_masks.shape[0]),
                    mode="constant",
                    value=0,
                )
            else:
                _domain_ids = _domain_ids[:1024]  # [1024, hidden_size]
                _domain_masks = _domain_masks[:1024]  # [1024]

            domain_ids_padded.append(_domain_ids)
            domain_masks_padded.append(_domain_masks)

        # 第六步：stack并投影
        new_domain_ids = torch.stack(domain_ids_padded)  # [bs, 1024, hidden_size]
        new_domain_masks = torch.stack(domain_masks_padded)  # [bs, 1024]

        encoder_out = self.esm_encoder(
            new_domain_ids, new_domain_masks
        ).last_hidden_state
        domain_feats = self.domain_feats_projector(
            encoder_out
        )  # [bs, 1024, gpt_hidden_size]

        return domain_feats, new_domain_masks

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
        assert (
            encoder_hidden_states is not None and encoder_attention_mask is not None
        ), "encoder_hidden_states and encoder_attention_mask are required"

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": encoder_attention_mask,
            }
        )

        return model_inputs

    def initialize_output_tokens(self, bs: int, **kwargs):
        start_id = self.tokenizer.cls_token_id
        input_ids = (
            (torch.zeros((1)) + start_id).unsqueeze(0).repeat(bs, 1)
        )  # create batch dim
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
            processed_string = processed_string.replace(" ", "")
            cleaned_data.append(processed_string)
        return cleaned_data

    def generate(
        self,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
        generation_config: GenerationConfig,
        **kwargs,
    ):
        # override the prepare_inputs_for_generation to use the cross attention
        self.gpt2.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        # 0) encoding
        encoder_out, encoder_mask = self.forward_encoder(
            domain_ids, domain_masks, num_domains_per_protein
        )

        # 1) initialized from all mask tokens
        initial_output_tokens = self.initialize_output_tokens(bs=encoder_out.shape[0])
        sample_results = self.gpt2.generate(
            initial_output_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
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


if __name__ == "__main__":
    from loguru import logger

    model = GPTwCatDomainConditioning(logger=logger)
    model.to("cuda")
    model.eval()
    batch = torch.load("analysis/assets/first_batch.pt")
    with torch.inference_mode():
        print(model(**batch))
