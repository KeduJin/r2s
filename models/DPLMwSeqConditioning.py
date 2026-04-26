from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import EsmModel

from utils.init_utils import construct_class_by_name
from utils.metrics import Scalar

from .BaseModel import BaseModel
from .DPLM.model import DiffusionProteinLanguageModel as DPLM


class DPLMwSeqConditioning(BaseModel):
    def __init__(
        self,
        dplm_type="airkingbd/dplm_150m",
        criterion_kwargs: Optional[dict] = None,
        weighting: str = "linear",
        **kwargs,
    ):
        # raise NotImplementedError("DPLMwSeqConditioning is not implemented yet")
        super().__init__(**kwargs)
        self.logger.info(
            f"DPLMwSeqConditioning initialized with dplm_type: {dplm_type}"
        )

        # Since we need to use the cross attention, we need to set the use_cache to False and is_decoder to True
        self.dplm = DPLM.from_pretrained(
            dplm_type,
            net_override={
                "add_cross_attention": True,
                "use_cache": False,
                "is_decoder": True,
            },
            cfg_override={"gradient_ckpt": True},
        )
        self.esm_encoder = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")

        self.esm_encoder.encoder.gradient_checkpointing = True
        self.esm_encoder.gradient_checkpointing_enable()

        # freeze the pooler and contact_head of esm_encoder to avoid the runtime error
        self.dplm.net.esm.contact_head.regression.weight.requires_grad = False
        self.dplm.net.esm.contact_head.regression.bias.requires_grad = False
        self.dplm.net.esm.embeddings.position_embeddings.weight.requires_grad = False
        self.esm_encoder.pooler.dense.weight.requires_grad = False
        self.esm_encoder.pooler.dense.bias.requires_grad = False
        self.esm_encoder.contact_head.regression.weight.requires_grad = False
        self.esm_encoder.contact_head.regression.bias.requires_grad = False
        self.esm_encoder.embeddings.position_embeddings.weight.requires_grad = False

        self.domain_feats_projector = nn.Linear(
            self.esm_encoder.config.hidden_size, self.dplm.net.config.hidden_size
        )
        self.tokenizer = self.dplm.tokenizer
        self.cfg = self.dplm.cfg

        # Zero-out cross-attention weights
        # for name, param in self.dplm.net.named_parameters():
        #     if "crossattention" in name:
        #         param.data.zero_()

        # for layer in self.dplm.net.esm.encoder.layer:
        #     if hasattr(layer, "crossattention"):
        #         torch.nn.init.zeros_(layer.crossattention.LayerNorm.bias)
        #         torch.nn.init.ones_(
        #             layer.crossattention.LayerNorm.weight
        #         )  # weight init to 1

        self.weighting = weighting
        if criterion_kwargs is not None:
            self.criterion = construct_class_by_name(**criterion_kwargs)
        else:
            self.criterion = None

    def set_objective_and_metrics(self, experiment, stage: str = "train"):
        # we have loss, nll_loss, ppl, fullseq_loss, fullseq_nll_loss, bsz, sample_size, sample_ratio, nonpad_ratio, weight_diff_loss
        train_metrics = None
        val_metrics = None
        test_metrics = None
        if stage == "train":
            train_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "nll_loss": Scalar(dist_sync_on_step=True),
                "ppl": Scalar(dist_sync_on_step=True),
                "fullseq_loss": Scalar(dist_sync_on_step=True),
                "fullseq_nll_loss": Scalar(dist_sync_on_step=True),
                "weight_diff_loss": Scalar(dist_sync_on_step=True),
                "sample_ratio": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "nll_loss": Scalar(dist_sync_on_step=True),
                "ppl": Scalar(dist_sync_on_step=True),
                "fullseq_loss": Scalar(dist_sync_on_step=True),
                "fullseq_nll_loss": Scalar(dist_sync_on_step=True),
                "weight_diff_loss": Scalar(dist_sync_on_step=True),
                "sample_ratio": Scalar(dist_sync_on_step=True),
            }
            train_metrics = {
                k: v.to(experiment.device) for k, v in train_metrics.items()
            }
            val_metrics = {k: v.to(experiment.device) for k, v in val_metrics.items()}
        else:
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

    def _forward(
        self, seq_ids: torch.Tensor, unmask_seq_ids: torch.Tensor, **kwargs
    ) -> dict:
        unmask_seq_emb, unmask_seq_masks = self.forward_encoder(unmask_seq_ids)

        logits = self.dplm(
            input_ids=seq_ids,
            cross_attention=unmask_seq_emb,
            cross_attention_mask=unmask_seq_masks,
        )
        return logits  # [bs, seq_len, vocab_size]

    def forward(
        self,
        seq_ids: torch.Tensor,
        **kwargs,
    ):
        # this function is used for training, computing the loss
        target = seq_ids

        t1, t2 = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (2 * target.size(0),),
            device=target.device,
        ).chunk(2)

        if self.cfg.rdm_couple:
            # couple training
            # refer to Appendix G: Improved Training with Conditioning
            # and Algorithm 3 in Zheng et al., 2023 (https://arxiv.org/pdf/2302.05737)
            x_t, t, loss_mask = list(
                self.dplm.q_sample_coupled(
                    target,
                    t1,
                    t2,
                    maskable_mask=self.dplm.get_non_special_symbol_mask(target),
                ).values()
            )
            target = target.repeat(2, 1)
        else:
            x_t, t, loss_mask = list(
                self.dplm.q_sample(
                    target,
                    t1,
                    maskable_mask=self.dplm.get_non_special_symbol_mask(target),
                ).values()
            )

        logits = self._forward(x_t, seq_ids)

        num_timesteps = self.cfg.num_diffusion_timesteps
        weight = {
            "linear": num_timesteps - (t - 1),
            "constant": num_timesteps * torch.ones_like(t),
        }[self.weighting][:, None].float() / num_timesteps
        if self.criterion is not None:
            # target_to_compute_loss = target.masked_fill(target == self.tokenizer.pad_token_id, -100)
            loss, logging_output = self.criterion(
                logits,
                target,
                loss_mask,
                weight,
                watch_t1_t2_loss=kwargs.get("watch_t1_t2_loss", False),
                cal_constant_loss=kwargs.get("cal_constant_loss", False),
            )
        else:
            loss, logging_output = None, {}
        return {
            "logits": logits,
            "target": target,
            "loss_mask": loss_mask,
            "weight": weight,
            "loss": loss,
            **logging_output,
        }

    def initialize_output_tokens(self, tokens, partial_masks=None, **kwargs):
        if tokens is None:
            raise NotImplementedError
        else:
            output_mask = self.dplm.get_non_special_symbol_mask(
                tokens, partial_masks=partial_masks
            )

            output_tokens = tokens.masked_fill(output_mask, self.dplm.mask_id)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores

    def generate(
        self,
        seq_ids: torch.Tensor,
        max_iter=None,
        temperature=None,
        partial_masks=None,
        sampling_strategy="gumbel_argmax",
        disable_resample=False,
        resample_ratio=0.25,
        verbose=True,
        **kwargs,
    ):
        # 0) encoding
        encoder_out, encoder_mask = self.forward_encoder(seq_ids)
        # 1) initialized from all mask tokens
        (
            initial_output_tokens,
            initial_output_scores,
        ) = self.initialize_output_tokens(seq_ids, partial_masks=partial_masks)

        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )

        prev_decoder_out["output_masks"] = self.dplm.get_non_special_symbol_mask(
            prev_decoder_out["output_tokens"], partial_masks=partial_masks
        )

        for step in tqdm(range(max_iter), desc="Decoding", disable=not verbose):
            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.dplm.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    encoder_out=encoder_out,
                    encoder_mask=encoder_mask,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    disable_resample=disable_resample,
                    resample_ratio=resample_ratio,
                )

            output_tokens = decoder_out["output_tokens"]
            output_scores = decoder_out["output_scores"]

            # 2.2: re-mask skeptical parts of low confidence
            non_special_sym_mask = self.dplm.get_non_special_symbol_mask(
                prev_decoder_out["output_tokens"], partial_masks=partial_masks
            )

            (
                output_masks,
                result_tokens,
                result_scores,
            ) = self.dplm._reparam_decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy="reparam-uncond-deterministic-linear",
                xt_neq_x0=prev_decoder_out["output_masks"],
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
                noise=self.dplm.mask_id,
            )

            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out["history"],
            )

        decoder_out = prev_decoder_out
        seq_list = self.decode_to_seq(decoder_out["output_tokens"])
        return {
            "output_tokens": decoder_out["output_tokens"],
            "output_scores": decoder_out["output_scores"],
            "output_seqs": seq_list,
        }

    def decode_to_seq(self, tokens):
        return [
            "".join(seq.split(" "))
            for seq in self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        ]
