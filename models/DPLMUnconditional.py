# from transformers import EsmModel
# we use dplm's implemented EsmModel, which use sdpa attention
# from byprot.models.dplm.modules.dplm_modeling_esm import ModifiedEsmModel as EsmModel
# or we can use flash attention
from typing import Optional

import torch
from byprot.models.dplm.modules.dplm_modeling_esm import EsmForDPLM
from tqdm import tqdm
from transformers import AutoConfig

from utils.init_utils import construct_class_by_name
from utils.metrics import Scalar

from .BaseModel import BaseModel
from .DPLM.model import DiffusionProteinLanguageModel as DPLM
from .ModifiedESM.modeling_modified_faesm import EsmModel as EsmModel


class DPLMUnconditional(BaseModel):
    def __init__(
        self,
        dplm_type="airkingbd/dplm_150m",
        criterion_kwargs: Optional[dict] = None,
        weighting: str = "linear",
        sampling_strategy: str = "gumbel_argmax",
        sampling_max_iter: int = 250,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger.info(f"DPLMUnconditional initialized with dplm_type: {dplm_type}")

        # Since we need to use the cross attention, we need to set the use_cache to False and is_decoder to True
        cfg_override = {"gradient_ckpt": True}
        net_override = {
            # "add_cross_attention": True,
            "use_cache": False,
            # "is_decoder": True,
        }
        if kwargs.get("dplm_pretrain", True):
            self.logger.info("DPLM pretrain is enabled, using pretrained DPLM model")
            self.dplm = DPLM.from_pretrained(
                dplm_type, net_override=net_override, cfg_override=cfg_override
            )
        else:
            self.logger.info(
                "DPLM pretrain is disabled, using random initialized DPLM model"
            )
            config = AutoConfig.from_pretrained(dplm_type, **net_override)
            net = EsmForDPLM(config)
            self.dplm = DPLM(cfg_override, net=net)

        # freeze the pooler and contact_head of esm_encoder to avoid the runtime error
        self.dplm.net.esm.contact_head.regression.weight.requires_grad = False
        self.dplm.net.esm.contact_head.regression.bias.requires_grad = False
        self.dplm.net.esm.embeddings.position_embeddings.weight.requires_grad = False

        self.tokenizer = self.dplm.tokenizer
        self.cfg = self.dplm.cfg

        self.weighting = weighting
        self.sampling_strategy = sampling_strategy
        self.sampling_max_iter = sampling_max_iter
        self.temperature = temperature
        if criterion_kwargs is not None:
            self.criterion = construct_class_by_name(**criterion_kwargs)
            self.logger.info(f"Criterion initialized with: {criterion_kwargs}")
        else:
            self.criterion = None

    def set_objective_and_metrics(self, stage: str = "train", experiment=None):
        # we have loss, nll_loss, ppl, fullseq_loss, fullseq_nll_loss, bsz, sample_size, sample_ratio, nonpad_ratio, weight_diff_loss
        if stage == "train":
            train_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "ppl": Scalar(dist_sync_on_step=True),
                "fullseq_loss": Scalar(dist_sync_on_step=True),
                "weight_diff_loss": Scalar(dist_sync_on_step=True),
                "sample_ratio": Scalar(dist_sync_on_step=True),
                "nonpad_ratio": Scalar(dist_sync_on_step=True),
                "sample_size": Scalar(dist_sync_on_step=True),
                "bsz": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "ppl": Scalar(dist_sync_on_step=True),
                "fullseq_loss": Scalar(dist_sync_on_step=True),
                "weight_diff_loss": Scalar(dist_sync_on_step=True),
                "sample_ratio": Scalar(dist_sync_on_step=True),
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
        **kwargs,
    ) -> dict:
        logits = self.dplm(
            input_ids=seq_ids,
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
            raise NotImplementedError("Coupled training is not used.")
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

        logits = self._forward(x_t)

        num_timesteps = self.cfg.num_diffusion_timesteps
        weight = {
            "linear": (
                num_timesteps - (t - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t),
        }[self.weighting][:, None].float() / num_timesteps

        loss, logging_output = None, {}
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
        partial_masks=None,
        disable_resample=True,
        resample_ratio=0.25,
        verbose=False,
        **kwargs,
    ):
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
            max_step=self.sampling_max_iter,
            history=[initial_output_tokens.clone()],
            temperature=self.temperature,
        )

        prev_decoder_out["output_masks"] = self.dplm.get_non_special_symbol_mask(
            prev_decoder_out["output_tokens"], partial_masks=partial_masks
        )

        for step in tqdm(
            range(self.sampling_max_iter), desc="Decoding", disable=not verbose
        ):
            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.dplm.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=self.sampling_strategy,
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
                max_step=self.sampling_max_iter,
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
