# from transformers import EsmModel
# we use dplm's implemented EsmModel, which use sdpa attention
# from byprot.models.dplm.modules.dplm_modeling_esm import ModifiedEsmModel as EsmModel
# or we can use flash attention
from typing import Optional

import torch
import torch.nn as nn
from byprot.models.dplm.modules.dplm_modeling_esm import EsmForDPLM
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoConfig

from utils.init_utils import construct_class_by_name
from utils.metrics import Scalar

from .BaseModel import BaseModel
from .DPLM.model import DiffusionProteinLanguageModel as DPLM
from .ModifiedESM.modeling_modified_faesm import EsmModel as EsmModel


class DPLMwDomainConditioning(BaseModel):
    def __init__(
        self,
        dplm_type="airkingbd/dplm_150m",
        criterion_kwargs: Optional[dict] = None,
        weighting: str = "linear",
        sampling_strategy: str = "gumbel_argmax",
        sampling_max_iter: int = 250,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger.info(
            f"DPLMwDomainConditioning initialized with dplm_type: {dplm_type}"
        )

        # Since we need to use the cross attention, we need to set the use_cache to False and is_decoder to True
        cfg_override = {"gradient_ckpt": True}
        net_override = {
            "add_cross_attention": True,
            "use_cache": False,
            "is_decoder": True,
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

        # Freeze DPLM pretrained parameters except cross attention
        # This is controlled by a flag in kwargs
        if kwargs.get("freeze_dplm_pretrain", False):
            if not kwargs.get("dplm_pretrain", True):
                raise ValueError(
                    "freeze_dplm_pretrain is only supported when dplm_pretrain is True"
                )
            self.logger.info(
                "Freezing DPLM pretrained parameters (except cross attention)..."
            )
            self._freeze_dplm_except_crossattention()
            self.logger.info("DPLM pretrained parameters frozen successfully.")
        else:
            self.logger.info("DPLM pretrained parameters are trainable.")

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
        #         )  # weight通常初始化为1

        self.weighting = weighting
        self.sampling_strategy = sampling_strategy
        self.sampling_max_iter = sampling_max_iter
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
                "domain_loss": Scalar(dist_sync_on_step=True),
                "non_domain_loss": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "ppl": Scalar(dist_sync_on_step=True),
                "fullseq_loss": Scalar(dist_sync_on_step=True),
                "weight_diff_loss": Scalar(dist_sync_on_step=True),
                "sample_ratio": Scalar(dist_sync_on_step=True),
                "domain_loss": Scalar(dist_sync_on_step=True),
                "non_domain_loss": Scalar(dist_sync_on_step=True),
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
        **kwargs,
    ) -> dict:
        domain_feats, domain_feat_masks = self.forward_encoder(
            domain_ids, domain_masks, num_domains_per_protein
        )

        logits = self.dplm(
            input_ids=seq_ids,
            cross_attention=domain_feats,
            cross_attention_mask=domain_feat_masks,
        )
        return logits  # [bs, seq_len, vocab_size]

    def forward(
        self,
        seq_ids: torch.Tensor,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
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

        logits = self._forward(x_t, domain_ids, domain_masks, num_domains_per_protein)

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
            domain_mask = self._create_domain_mask(seq_ids, kwargs["domain_positions"])
            loss, logging_output = self.criterion(
                logits,
                target,
                loss_mask,
                weight,
                watch_t1_t2_loss=kwargs.get("watch_t1_t2_loss", False),
                cal_constant_loss=kwargs.get("cal_constant_loss", False),
                domain_mask=domain_mask,
            )
        return {
            "logits": logits,
            "target": target,
            "loss_mask": loss_mask,
            "weight": weight,
            "loss": loss,
            **logging_output,
        }

    def _create_domain_mask(
        self,
        input_ids: torch.Tensor,
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
        batch_size, seq_len = input_ids.shape
        domain_mask = torch.zeros(batch_size, seq_len, device=input_ids.device)

        try:
            cls_token_id = self.tokenizer.cls_token_id
        except AttributeError:
            raise ValueError(
                "Could not find cls_token_id in model config. Please ensure it's available."
            )

        for i, positions in enumerate(domain_positions):
            # noted that here we have domain token in the input_ids, so we need to shift to seq ids
            cls_indices = (input_ids[i] == cls_token_id).nonzero(as_tuple=True)[0]
            if cls_indices.numel() == 0:
                # if we cannot find the cls token in the input_ids, skip this sample
                continue
            seq_start_offset = cls_indices[0].item()
            for start, end in positions:
                domain_mask[
                    i, start + seq_start_offset + 1 : end + seq_start_offset + 1
                ] = 1  # shift by 1 to exclude the cls token

        return domain_mask

    def forward_encoder(
        self,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
    ):
        encoder_out = self.esm_encoder(domain_ids, domain_masks).last_hidden_state
        encoder_emb_dim = encoder_out.shape[-1]

        # 第四步：按照原始结构重新组织数据
        domain_feats = []
        domain_feat_masks = []

        start_idx = 0
        for i, domain_count in enumerate(num_domains_per_protein):
            end_idx = start_idx + domain_count

            # 提取当前data point的encoder输出
            # [num_domain, seq_len, hidden_size]
            current_encoder_out = encoder_out[start_idx:end_idx]
            # [num_domain, seq_len]
            current_domain_masks = domain_masks[start_idx:end_idx]

            # 重塑并去除padding
            # [num_domain*seq_len, hidden_size]
            encoder_out_flat = current_encoder_out.reshape(-1, encoder_emb_dim)
            # [num_domain*seq_len]
            domain_masks_flat = current_domain_masks.reshape(-1)

            # 去除padding
            bool_mask = domain_masks_flat.bool()
            encoder_out_flat = encoder_out_flat[bool_mask]
            domain_masks_flat = domain_masks_flat[bool_mask]

            domain_feats.append(encoder_out_flat)
            domain_feat_masks.append(domain_masks_flat)

            start_idx = end_idx

        # 第五步：重新padding到1024长度
        domain_feats_padded = []
        domain_feat_masks_padded = []

        for domain_feat, domain_feat_mask in zip(domain_feats, domain_feat_masks):
            if domain_feat.shape[0] < 1024:
                domain_feat = F.pad(
                    domain_feat,
                    (0, 0, 0, 1024 - domain_feat.shape[0]),
                    mode="constant",
                    value=0,
                )
                domain_feat_mask = F.pad(
                    domain_feat_mask,
                    (0, 1024 - domain_feat_mask.shape[0]),
                    mode="constant",
                    value=0,
                )
            else:
                domain_feat = domain_feat[:1024, :]  # [1024, hidden_size]
                domain_feat_mask = domain_feat_mask[:1024]  # [1024]

            domain_feats_padded.append(domain_feat)
            domain_feat_masks_padded.append(domain_feat_mask)

        # 第六步：stack并投影
        # [bs, 1024, hidden_size]
        domain_feats = torch.stack(domain_feats_padded)
        domain_feats = self.domain_feats_projector(
            domain_feats
        )  # [bs, 1024, gpt_hidden_size]
        domain_feat_masks = torch.stack(domain_feat_masks_padded)  # [bs, 1024]

        return domain_feats, domain_feat_masks

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
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
        temperature=None,
        partial_masks=None,
        disable_resample=True,
        resample_ratio=0.25,
        verbose=False,
        **kwargs,
    ):
        # 0) encoding
        encoder_out, encoder_mask = self.forward_encoder(
            domain_ids, domain_masks, num_domains_per_protein
        )
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
            temperature=temperature,
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
                    encoder_out=encoder_out,
                    encoder_mask=encoder_mask,
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

    def _freeze_dplm_except_crossattention(self):
        """
        Freeze all DPLM pretrained parameters except cross attention layers.
        This allows only cross attention to learn while keeping pretrained weights fixed.
        """
        # First, freeze all DPLM parameters
        for name, param in self.dplm.net.named_parameters():
            param.requires_grad = False

        # Then, unfreeze cross attention parameters
        cross_attn_count = 0
        for name, param in self.dplm.net.named_parameters():
            if "crossattention" in name.lower():
                param.requires_grad = True
                cross_attn_count += 1
                # self.logger.debug(f"Unfrozen cross attention param: {name}")

        # Also unfreeze the LM head if it exists (for output projection)
        # Actually, we should keep LM head frozen if we want to freeze pretrained params
        # But let's check if there's a separate head

        # Log statistics
        total_params = sum(p.numel() for p in self.dplm.net.parameters())
        trainable_params = sum(
            p.numel() for p in self.dplm.net.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        self.logger.info(
            f"Cross attention parameters unfrozen: {cross_attn_count} parameter groups"
        )
        self.logger.info(
            f"DPLM parameter statistics - Total: {total_params:,}, "
            f"Trainable: {trainable_params:,}, Frozen: {frozen_params:,}"
        )

    def get_param_groups(
        self, pretrain_lr: float = 1e-5, new_init_lr: float = 1e-4, **optimizer_kwargs
    ):
        """
        返回参数组，用于设置不同的学习率。

        Args:
            pretrain_lr: DPLM预训练参数的学习率，默认 1e-5
            new_init_lr: 新初始化参数的学习率，默认 1e-4
            **optimizer_kwargs: 优化器的其他参数（如 weight_decay, betas 等）

        Returns:
            param_groups: 包含两个参数组的列表
                - param_groups[0]: DPLM预训练参数（学习率 pretrain_lr）
                - param_groups[1]: 新初始化参数（学习率 new_init_lr）
        """
        dplm_pretrain_params = []
        dplm_cross_attn_params = []

        for name, param in self.dplm.net.named_parameters():
            if param.requires_grad:
                if "crossattention" in name.lower():
                    dplm_cross_attn_params.append(param)
                else:
                    dplm_pretrain_params.append(param)

        new_init_params = []
        new_init_params.extend(list(self.domain_feats_projector.parameters()))
        new_init_params.extend(
            [p for p in self.esm_encoder.parameters() if p.requires_grad]
        )
        new_init_params.extend(dplm_cross_attn_params)

        param_groups = []

        if len(dplm_pretrain_params) > 0:
            param_groups.append(
                {
                    "params": dplm_pretrain_params,
                    "lr": pretrain_lr,
                    **{k: v for k, v in optimizer_kwargs.items() if k != "lr"},
                }
            )
            self.logger.info(
                f"DPLM pretrain parameter group: {len(dplm_pretrain_params)} parameters, "
                f"lr={pretrain_lr}, total params={sum(p.numel() for p in dplm_pretrain_params):,}"
            )

        # 第二组：新初始化参数
        if len(new_init_params) > 0:
            param_groups.append(
                {
                    "params": new_init_params,
                    "lr": new_init_lr,
                    **{k: v for k, v in optimizer_kwargs.items() if k != "lr"},
                }
            )
            self.logger.info(
                f"New init parameter group: {len(new_init_params)} parameters, "
                f"lr={new_init_lr}, total params={sum(p.numel() for p in new_init_params):,}"
            )

        return param_groups


if __name__ == "__main__":
    import logging

    logger = logging.getLogger(__name__)
    model = DPLMwDomainConditioning(dplm_type="airkingbd/dplm_150m", logger=logger)
    # model.forward_encoder(domain_ids=torch.randint(0, 100, (10, 1024)), domain_masks=torch.randint(0, 1, (10, 1024)), num_domains_per_protein=torch.randint(1, 10, (10,)))
