import re

import torch
import torch.nn as nn
from transformers import (
    EsmModel,
    EsmTokenizer,
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
)

from utils.metrics import Scalar

from .BaseModel import BaseModel


class GPTwSeqConditioning(BaseModel):
    def __init__(
        self,
        gpt_type="gpt2-amino_acid",
        esm_type="facebook/esm2_t12_35M_UR50D",
        **kwargs,
    ):
        super().__init__(**kwargs)
        raise NotImplementedError("GPTwSeqConditioning is not implemented yet")
        self.logger.info(f"GPTwSeqConditioning initialized with gpt_type: {gpt_type}")

        self.tokenizer = EsmTokenizer.from_pretrained("airkingbd/dplm_150m")
        # Since we need to use the cross attention, we need to set the use_cache to False and is_decoder to True
        gpt2_config = GPT2Config.from_pretrained(
            gpt_type,
            add_cross_attention=True,
            cache_dir="/storage/yuanfajieLab/yuanfajie/my_project/checkpoints",
            local_files_only=True,
            vocab_size=len(self.tokenizer.get_vocab()),
        )
        self.gpt2 = GPT2LMHeadModel(gpt2_config)
        # self.gpt2.transformer.gradient_checkpointing = True
        self.gpt2.gradient_checkpointing_enable()
        self.esm_encoder = EsmModel.from_pretrained(esm_type)

        # freeze the pooler and contact_head of esm_encoder to avoid the runtime error
        # self.dplm.net.esm.contact_head.regression.weight.requires_grad = False
        # self.dplm.net.esm.contact_head.regression.bias.requires_grad = False
        # self.dplm.net.esm.embeddings.position_embeddings.weight.requires_grad = False

        self.esm_encoder.pooler.dense.weight.requires_grad = False
        self.esm_encoder.pooler.dense.bias.requires_grad = False
        self.esm_encoder.contact_head.regression.weight.requires_grad = False
        self.esm_encoder.contact_head.regression.bias.requires_grad = False
        self.esm_encoder.embeddings.position_embeddings.weight.requires_grad = False

        self.domain_feats_projector = nn.Linear(
            self.esm_encoder.config.hidden_size, self.gpt2.config.hidden_size
        )
        # self.cfg = self.dplm.cfg

    def set_objective_and_metrics(self, stage: str = "train", experiment=None):
        # we have loss, nll_loss, ppl, fullseq_loss, fullseq_nll_loss, bsz, sample_size, sample_ratio, nonpad_ratio, weight_diff_loss
        if stage == "train":
            train_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
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

    def _forward(self, seq_ids: torch.Tensor, labels: torch.Tensor, **kwargs) -> dict:
        unmask_seq_emb, unmask_seq_masks = self.forward_encoder(seq_ids)
        out_dict = self.gpt2(
            input_ids=seq_ids,
            encoder_hidden_states=unmask_seq_emb,
            encoder_attention_mask=unmask_seq_masks,
            labels=labels,
            return_dict=True,
        )
        return out_dict  # [bs, seq_len, vocab_size]

    def forward(self, seq_ids: torch.Tensor, **kwargs):
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

        out_dict = self._forward(seq_ids, labels=target)
        logits = out_dict["logits"]
        loss = out_dict["loss"]
        return {"logits": logits, "target": target, "loss": loss}

    def forward_encoder(
        self,
        unmask_seq_ids: torch.Tensor,
    ):
        unmask_seq_masks = unmask_seq_ids.ne(self.tokenizer.pad_token_id)
        encoder_out = self.esm_encoder(
            unmask_seq_ids, unmask_seq_masks
        ).last_hidden_state
        encoder_out = self.domain_feats_projector(
            encoder_out
        )  # [bs, seq_len, hidden_size of dplm]

        return encoder_out, unmask_seq_masks

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
        seq_ids: torch.Tensor,
        generation_config: GenerationConfig,
        verbose=True,
        **kwargs,
    ):
        # override the prepare_inputs_for_generation to use the cross attention
        self.gpt2.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        # 0) encoding
        encoder_out, encoder_mask = self.forward_encoder(seq_ids)

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

    model = GPTwSeqConditioning(logger=logger)
    model.to("cuda")
    model.eval()
