import re
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import EsmTokenizer, GenerationConfig

from utils.init_utils import construct_class_by_name
from utils.metrics import Scalar

from .BaseModel import BaseModel
from .Qwen3.configuration_domainconditioning_qwen3 import Qwen3Config
from .Qwen3.modeling_domainconditioning_qwen3 import Qwen3CAForCausalLM
from .ModifiedESM.modeling_modified_faesm import EsmModel as EsmModel


## Attention, We do not encode the domain directly, instead we encode domain pieces (sub-domains) separately.
## so do not use num_domains_per_protein, use num_domain_pieces_per_protein instead.
## Unified version: Use the same encoder for both domain encoding (for cross-attention) and domain piece encoding (for extended vocabulary)
class RAGQwen3CAwDomainConditioningUnified(BaseModel):
    def __init__(
        self,
        qwen3_type="Qwen/Qwen3-100M",
        esm_type="facebook/esm2_t12_35M_UR50D",
        criterion_kwargs: Optional[dict] = None,
        dynamic_domain_weight: bool = False,
        dynamic_domain_weight_p_rate: float = 1,
        disable_domain_aa_mask: bool = False, # if False, we will mask the domain aa with -100 in target, except the first aa (which should be replaced by the domain piece token id)
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.logger.info(
            f"{self.__class__.__name__} initialized with qwen3_type: {qwen3_type}, esm_type: {esm_type} (unified encoder)"
        )
        self.dynamic_domain_weight = dynamic_domain_weight
        self.dynamic_domain_weight_p_rate = dynamic_domain_weight_p_rate
        self.disable_domain_aa_mask = disable_domain_aa_mask
        self.tokenizer = EsmTokenizer.from_pretrained("airkingbd/dplm_150m")
        # Since we need to use the cross attention, we need to set the use_cache to False and is_decoder to True
        if qwen3_type == "Qwen/Qwen3-100M":
            # 创建100M配置
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
                use_cache=True,  #
                use_sliding_window=False,
                vocab_size=len(self.tokenizer.get_vocab()),
                add_cross_attention=True,
            )
            # 创建模型
            self.qwen3 = Qwen3CAForCausalLM(config)
        else:
            config = Qwen3Config.from_pretrained(
                qwen3_type,
                add_cross_attention=True,
                vocab_size=len(self.tokenizer.get_vocab()),
            )
            self.qwen3 = Qwen3CAForCausalLM(config)

        self.qwen3.gradient_checkpointing_enable()

        self.esm_encoder = EsmModel.from_pretrained(esm_type)
        self.esm_encoder.encoder.gradient_checkpointing = True
        self.esm_encoder.gradient_checkpointing_enable()

        self.esm_encoder.pooler.dense.weight.requires_grad = False
        self.esm_encoder.pooler.dense.bias.requires_grad = False
        self.esm_encoder.contact_head.regression.weight.requires_grad = False
        self.esm_encoder.contact_head.regression.bias.requires_grad = False
        # self.esm_encoder.embeddings.position_embeddings.weight.requires_grad = False

        # Unified: 使用同一个projector，因为domain和phrase都来自同一个encoder
        self.unified_projector = nn.Linear(
            self.esm_encoder.config.hidden_size, self.qwen3.config.hidden_size
        )

        # Domain order embeddings: 用于标识每个domain piece在蛋白质中的顺序
        max_domain_pieces = 128  # 假设每个蛋白质最多128个domain pieces
        self.domain_order_embeddings = nn.Embedding(
            max_domain_pieces, self.esm_encoder.config.hidden_size
        )

        self.non_vector = torch.zeros(1, self.qwen3.config.hidden_size)

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
                "cls_head_l2_norm_mean": Scalar(dist_sync_on_step=True),
                "cls_head_l2_norm_std": Scalar(dist_sync_on_step=True),
                "domain_head_l2_norm_mean": Scalar(dist_sync_on_step=True),
                "domain_head_l2_norm_std": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "domain_loss": Scalar(dist_sync_on_step=True),
                "non_domain_loss": Scalar(dist_sync_on_step=True),
                "domain_weight": Scalar(dist_sync_on_step=True),
                "non_domain_weight": Scalar(dist_sync_on_step=True),
                "unweighted_loss": Scalar(dist_sync_on_step=True),
                "cls_head_l2_norm_mean": Scalar(dist_sync_on_step=True),
                "cls_head_l2_norm_std": Scalar(dist_sync_on_step=True),
                "domain_head_l2_norm_mean": Scalar(dist_sync_on_step=True),
                "domain_head_l2_norm_std": Scalar(dist_sync_on_step=True),
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
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
        # labels: torch.Tensor,
        **kwargs,
    ) -> dict:
        # Unified: 使用同一个encoder编码所有domain pieces
        # 返回：用于cross-attention的特征 和 用于扩展词表的pooler_output
        domain_feats, domain_feat_masks, pooler_output = self.forward_unified_encoder(
            domain_pieces_ids,
            domain_pieces_masks,
            num_domain_pieces_per_protein
        )

        out_dict = self.qwen3(
            input_ids=seq_ids,
            encoder_hidden_states=domain_feats,
            encoder_attention_mask=domain_feat_masks,
            output_hidden_states=True,
            # labels=labels, # we calculate the loss handcraftly.
            return_dict=True,
        )

        # Unified: 使用同一个encoder编码domain pieces（用于扩展词表）
        phrase_out = self.unified_projector(
            pooler_output
        )  # [num_domain_pieces, hidden_size]

        return (
            out_dict,
            phrase_out,
        )  # [bs, seq_len, vocab_size], [num_domain_pieces, hidden_size]

    def forward(
        self,
        seq_ids: torch.Tensor,
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
        domain_positions: list[list[tuple[int, int]]],
        **kwargs,
    ):
        """
        this function is used for training, computing the loss
        seq_ids: [bs, seq_len]
        domain_pieces_ids: [num_all_pieces, piece_seq_len]
        domain_pieces_masks: [num_all_pieces, piece_seq_len]
        num_domain_pieces_per_protein: [bs]
        domain_positions: list of list of (start, end) tuples
        kwargs: other arguments

        return:
            logits: [bs, seq_len, vocab_size]
            target: [bs, seq_len]
            loss_mask: [bs, seq_len]
            loss: the loss value
        """
        target = seq_ids.clone()
        target = target.masked_fill(target == self.tokenizer.pad_token_id, -100)
        orig_target = target.clone()
        out_dict, phrase_out = self._forward(
            seq_ids,
            domain_pieces_ids,
            domain_pieces_masks,
            num_domain_pieces_per_protein,
        )
        target, domain_weight_matrix = self.create_rag_target(
            target, domain_positions, num_domain_pieces_per_protein, self.dynamic_domain_weight
        )
        loss_dict = self.rag_loss(
            out_dict, phrase_out, target, num_domain_pieces_per_protein, domain_weight_matrix
        )

        return {
            "target": target,
            "loss": loss_dict["loss"],
            "domain_loss": loss_dict["domain_loss"],
            "non_domain_loss": loss_dict["non_domain_loss"],
            "domain_weight": 1,
            "non_domain_weight": 1,
            "cls_head_l2_norm_mean": loss_dict["cls_head_l2_norm_mean"],
            "cls_head_l2_norm_std": loss_dict["cls_head_l2_norm_std"],
            "domain_head_l2_norm_mean": loss_dict["domain_head_l2_norm_mean"],
            "domain_head_l2_norm_std": loss_dict["domain_head_l2_norm_std"],
        }

    def create_rag_target(
        self, target, domain_positions, num_domain_pieces_per_protein, dynamic_domain_weight=False
    ):
        # 根据domain_positions修改target
        # Domain position 是0-based索引，左闭右开, 以domain piece为元
        # 如果当前aa在domain piece外面，target不变
        # 如果当前aa在domain里面：
        #   - 如果是domain piece的第一个aa，target改成这个domain piece在当前batch的idx (vocab_size + domain piece在phrase_out中的索引)
        #   - 如果是domain piece的其他aa，target改成-100（不计算loss）
        vocab_size = self.qwen3.config.vocab_size
        batch_size = target.size(0)
        seq_len = target.size(1)

        domain_weight_matrix = torch.ones_like(target, dtype=torch.float, device=target.device)

        # 计算每个样本的domain在phrase_out中的起始索引
        if len(num_domain_pieces_per_protein) > 1:
            domain_start_indices = torch.cumsum(
                torch.cat(
                    [
                        torch.tensor(
                            [0],
                            device=num_domain_pieces_per_protein.device,
                            dtype=num_domain_pieces_per_protein.dtype,
                        ),
                        num_domain_pieces_per_protein[:-1],
                    ]
                ),
                dim=0,
            )
        else:
            domain_start_indices = torch.tensor(
                [0],
                device=num_domain_pieces_per_protein.device,
                dtype=num_domain_pieces_per_protein.dtype,
            )

        for i, positions in enumerate(domain_positions):
            sample_domain_start_idx = domain_start_indices[i].item()
            for domain_idx_in_sample, (start, end) in enumerate(positions):
                domain_global_idx = sample_domain_start_idx + domain_idx_in_sample
                domain_token_id = vocab_size + domain_global_idx

                domain_start_pos = start + 1
                domain_end_pos = end + 1

                if domain_start_pos >= target.size(1):
                    continue

                target[i, domain_start_pos] = domain_token_id

                if not self.disable_domain_aa_mask and domain_start_pos + 1 < target.size(1):
                    actual_end_pos = min(
                        domain_end_pos, target.size(1)
                    )  # domain_end_pos是开区间
                    target[i, domain_start_pos + 1 : actual_end_pos] = -100
                
                if dynamic_domain_weight:
                    domain_length = end - start
                    domain_weight_matrix[i, domain_start_pos] = float(domain_length) * self.dynamic_domain_weight_p_rate
        return target, domain_weight_matrix

    def rag_loss(self, out_dict, phrase_out, target, num_domain_pieces_per_protein, domain_weight_matrix):
        hidden_states = out_dict["hidden_states"][-1]
        lm_head_weights = self.qwen3.lm_head.weight
        vocab_size = lm_head_weights.size(0)
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        # 验证vocab_size的一致性
        config_vocab_size = self.qwen3.config.vocab_size
        if vocab_size != config_vocab_size:
            # 如果不一致，使用实际的lm_head_weights.size(0)
            # 这可能在模型加载后vocab_size被修改时发生
            raise ValueError(
                f"vocab_size {vocab_size} does not match config_vocab_size {config_vocab_size}"
            )

        extended_lm_head_weights = torch.cat(
            [lm_head_weights, phrase_out], dim=0
        )  # [vocab_size+num_domains, hidden_size]

        logits = (
            hidden_states @ extended_lm_head_weights.T
        )  # [batch, seq_len, vocab_size+num_domains]

        # 优化：直接在logits上操作，避免创建大的mask tensor
        # 对于第i个样本，它的domain在phrase_out中的索引范围是：
        # start_idx = sum(num_domain_pieces_per_protein[:i])
        # end_idx = sum(num_domain_pieces_per_protein[:i+1])
        # 在extended词表中，这些domain的索引是 vocab_size + start_idx 到 vocab_size + end_idx - 1
        total_vocab_size = extended_lm_head_weights.size(0)

        if len(num_domain_pieces_per_protein) > 1:
            domain_start_indices = torch.cumsum(
                torch.cat(
                    [
                        torch.tensor(
                            [0],
                            device=num_domain_pieces_per_protein.device,
                            dtype=num_domain_pieces_per_protein.dtype,
                        ),
                        num_domain_pieces_per_protein[:-1],
                    ]
                ),
                dim=0,
            )
        else:
            domain_start_indices = torch.tensor(
                [0],
                device=num_domain_pieces_per_protein.device,
                dtype=num_domain_pieces_per_protein.dtype,
            )

        # 优化：直接修改logits，不创建完整的mask tensor
        mask_value = -1e10
        for i in range(batch_size):
            # 第i个样本可以使用的domain在phrase_out中的索引范围
            start_idx = domain_start_indices[i].item()
            end_idx = start_idx + num_domain_pieces_per_protein[i].item()

            # 在extended词表中，domain的索引是 vocab_size + start_idx 到 vocab_size + end_idx - 1
            domain_start_in_vocab = vocab_size + start_idx
            domain_end_in_vocab = vocab_size + end_idx

            # 对于第i个样本，mask掉不属于它的domain
            # mask掉前面的domain（属于其他样本的domain）
            if domain_start_in_vocab > vocab_size:
                logits[i, :, vocab_size:domain_start_in_vocab] = mask_value
            # mask掉后面的domain（属于其他样本的domain）
            if domain_end_in_vocab < total_vocab_size:
                logits[i, :, domain_end_in_vocab:] = mask_value

        shifted_logits = logits[:, :-1, :]
        flat_shifted_logits = shifted_logits.reshape(-1, shifted_logits.size(-1))
        shifted_target = target[:, 1:]
        flat_shifted_target = shifted_target.reshape(-1)

        shifted_domain_weight_matrix = domain_weight_matrix[:, 1:]
        flat_shifted_domain_weight_matrix = shifted_domain_weight_matrix.reshape(-1)

    
        domain_mask = (
            (flat_shifted_target >= vocab_size)
            & (flat_shifted_target < total_vocab_size)
            & (flat_shifted_target != -100)
        )
        non_domain_mask = (flat_shifted_target < vocab_size) & (
            flat_shifted_target != -100
        )

        # 计算总的loss（用于反向传播）
        loss = F.cross_entropy(
            flat_shifted_logits,
            flat_shifted_target,
            ignore_index=-100,
            reduction="none",  # 先不reduce，分别计算domain和非domain的loss
        )

        # shift the domain_weight_matrix by 1
        loss = loss * flat_shifted_domain_weight_matrix

        # 计算domain部分的loss（按token取平均，domain当一个token算）
        domain_loss = None
        if domain_mask.any():
            domain_loss = loss[domain_mask].mean()

        # 计算非domain部分的loss（按token取平均）
        non_domain_loss = None
        if non_domain_mask.any():
            non_domain_loss = loss[non_domain_mask].mean()

        # 计算总的平均loss（用于反向传播）
        # total_loss = loss.mean()
        total_token = ((flat_shifted_target != -100).float() * flat_shifted_domain_weight_matrix) # valid token * weight
        total_loss = loss.sum() / total_token.sum()
        return {
            "loss": total_loss,
            "domain_loss": domain_loss
            if domain_loss is not None
            else torch.tensor(0.0, device=total_loss.device),
            "non_domain_loss": non_domain_loss
            if non_domain_loss is not None
            else torch.tensor(0.0, device=total_loss.device),
            "cls_head_l2_norm_mean": lm_head_weights.norm(p=2, dim=1).mean(),
            "cls_head_l2_norm_std": lm_head_weights.norm(p=2, dim=1).std(),
            "domain_head_l2_norm_mean": phrase_out.norm(p=2, dim=1).mean(),
            "domain_head_l2_norm_std": phrase_out.norm(p=2, dim=1).std(),
        }

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

    def forward_unified_encoder(
        self,
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
    ):
        """
        统一的encoder：使用domain pieces作为输入

        Args:
            domain_pieces_ids: [num_all_pieces, piece_seq_len] 所有domain pieces的token ids
            domain_pieces_masks: [num_all_pieces, piece_seq_len] 所有domain pieces的attention mask
            num_domain_pieces_per_protein: [bs] 每个蛋白质的domain piece数量

        Returns:
            domain_feats_padded: [bs, target_len, qwen_hidden_size] 用于cross-attention
            domain_feat_masks_padded: [bs, target_len] attention mask
            pooler_output: [num_all_pieces, esm_hidden_size] 用于扩展词表
        """
        batch_size = len(num_domain_pieces_per_protein)
        num_all_pieces = domain_pieces_ids.size(0)

        # 1. 为每个domain piece添加order embeddings
        # 计算每个piece在其所属蛋白质中的顺序
        piece_orders = []
        for i in range(batch_size):
            num_pieces = num_domain_pieces_per_protein[i].item()
            piece_orders.extend(list(range(num_pieces)))
        piece_orders = torch.tensor(piece_orders, device=domain_pieces_ids.device)

        # 2. 编码所有domain pieces
        encoder_out = self.esm_encoder(
            domain_pieces_ids,
            domain_pieces_masks
        )
        last_hidden_state = encoder_out.last_hidden_state  # [num_all_pieces, piece_seq_len, hidden_size]
        pooler_output = encoder_out.pooler_output  # [num_all_pieces, hidden_size]

        # 3. 添加domain order embeddings到last_hidden_state
        order_emb = self.domain_order_embeddings(piece_orders)  # [num_all_pieces, hidden_size]
        last_hidden_state = last_hidden_state + order_emb.unsqueeze(1)  # broadcast到seq_len维度

        # 4. 按蛋白质重组：将同一蛋白质的所有pieces在seq_len维度cat起来
        encoder_emb_dim = last_hidden_state.shape[-1]

        # 使用torch.split按num_domain_pieces_per_protein分割
        pieces_list = torch.split(last_hidden_state, num_domain_pieces_per_protein.tolist())
        masks_list = torch.split(domain_pieces_masks, num_domain_pieces_per_protein.tolist())

        domain_feats = []
        domain_feat_masks = []
        max_valid_len = 0

        for pieces, masks in zip(pieces_list, masks_list):
            # pieces: [num_pieces_in_protein, piece_seq_len, hidden_size]
            # masks: [num_pieces_in_protein, piece_seq_len]

            # 去除padding并cat
            valid_tokens = []
            valid_masks = []

            for piece, mask in zip(pieces, masks):
                # piece: [piece_seq_len, hidden_size]
                # mask: [piece_seq_len]
                bool_mask = mask.bool()
                valid_token = piece[bool_mask]  # [valid_len, hidden_size]
                valid_mask = mask[bool_mask]  # [valid_len]

                valid_tokens.append(valid_token)
                valid_masks.append(valid_mask)

            # 在seq_len维度cat所有pieces
            cat_tokens = torch.cat(valid_tokens, dim=0)  # [total_valid_len, hidden_size]
            cat_masks = torch.cat(valid_masks, dim=0)  # [total_valid_len]

            domain_feats.append(cat_tokens)
            domain_feat_masks.append(cat_masks)

            max_valid_len = max(max_valid_len, cat_tokens.shape[0])

        # 5. Padding到统一长度
        target_len = min(max_valid_len, 1024)

        domain_feats_padded = torch.zeros(
            batch_size, target_len, encoder_emb_dim,
            dtype=last_hidden_state.dtype, device=last_hidden_state.device
        )
        domain_feat_masks_padded = torch.zeros(
            batch_size, target_len,
            dtype=domain_pieces_masks.dtype, device=domain_pieces_masks.device
        )

        for i, (feat, mask) in enumerate(zip(domain_feats, domain_feat_masks)):
            valid_len = min(feat.shape[0], target_len)
            domain_feats_padded[i, :valid_len] = feat[:valid_len]
            domain_feat_masks_padded[i, :valid_len] = mask[:valid_len]

        # 6. 投影到Qwen3的hidden size
        domain_feats_padded = self.unified_projector(
            domain_feats_padded
        )  # [bs, target_len, qwen_hidden_size]

        return domain_feats_padded, domain_feat_masks_padded, pooler_output

    def forward_encoder(
        self,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
    ):
        """
        旧版本的forward_encoder，保留用于兼容性
        现在应该使用forward_unified_encoder
        """
        encoder_out = self.esm_encoder(domain_ids, domain_masks)
        encoder_out, pooler_output = encoder_out.last_hidden_state, encoder_out.pooler_output
        encoder_emb_dim = encoder_out.shape[-1]
        batch_size = len(num_domains_per_protein)

        # 使用torch.split一次性分割，避免循环索引
        encoder_out_list = torch.split(encoder_out, num_domains_per_protein.tolist())
        domain_masks_list = torch.split(domain_masks, num_domains_per_protein.tolist())

        # 向量化处理：去除padding
        domain_feats = []
        domain_feat_masks = []
        max_valid_len = 0

        for enc_out, mask in zip(encoder_out_list, domain_masks_list):
            # 重塑并去除padding
            flat_out = enc_out.reshape(-1, encoder_emb_dim)  # [num_domain*seq_len, hidden_size]
            flat_mask = mask.reshape(-1)  # [num_domain*seq_len]

            # 使用bool索引去除padding
            bool_mask = flat_mask.bool()
            valid_out = flat_out[bool_mask]
            valid_mask = flat_mask[bool_mask]

            domain_feats.append(valid_out)
            domain_feat_masks.append(valid_mask)

            # 动态计算最大长度
            max_valid_len = max(max_valid_len, valid_out.shape[0])

        # 动态padding：根据batch内最大长度padding，而不是固定1024
        target_len = min(max_valid_len, 1024)

        # 批量padding操作
        domain_feats_padded = torch.zeros(
            batch_size, target_len, encoder_emb_dim,
            dtype=encoder_out.dtype, device=encoder_out.device
        )
        domain_feat_masks_padded = torch.zeros(
            batch_size, target_len,
            dtype=domain_masks.dtype, device=domain_masks.device
        )

        for i, (feat, mask) in enumerate(zip(domain_feats, domain_feat_masks)):
            valid_len = min(feat.shape[0], target_len)
            domain_feats_padded[i, :valid_len] = feat[:valid_len]
            domain_feat_masks_padded[i, :valid_len] = mask[:valid_len]

        # 投影
        domain_feats_padded = self.unified_projector(
            domain_feats_padded
        )  # [bs, target_len, qwen_hidden_size]

        return domain_feats_padded, domain_feat_masks_padded, pooler_output

    def forward_encoder_old(
        self,
        # domain_ids_list: list[torch.Tensor],
        # domain_masks_list: list[torch.Tensor]
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
    ):
        domain_ids_list = domain_ids.split(num_domains_per_protein.tolist())
        domain_masks_list = domain_masks.split(num_domains_per_protein.tolist())

        domain_feats = []
        domain_feat_masks = []
        for domain_ids, domain_masks in zip(domain_ids_list, domain_masks_list):
            encoder_out = self.esm_encoder(domain_ids, domain_masks).last_hidden_state
            encoder_emb_dim = encoder_out.shape[-1]

            encoder_out = encoder_out.reshape(
                -1, encoder_emb_dim
            )  # [num_domain, seq_len, hidden_size of esm] to [num_domain*seq_len, hidden_size of esm]
            domain_masks = domain_masks.reshape(
                -1
            )  # [num_domain, seq_len] to [num_domain*seq_len]
            # now we want to remove the padding in the encoder_out and domain_masks
            bool_mask = domain_masks.bool()
            encoder_out = encoder_out[bool_mask]
            domain_masks = domain_masks[bool_mask]

            domain_feats.append(encoder_out)
            domain_feat_masks.append(domain_masks)

        # now we re-pad the domain_feats and domain_feat_masks to the 1024 length
        # so we can stack them together
        domain_feats_padded = []
        domain_feat_masks_padded = []
        for domain_feat, domain_feat_mask in zip(domain_feats, domain_feat_masks):
            # domain_feat: [num_domain*seq_len, hidden_size of esm]
            # domain_feat_mask: [num_domain*seq_len]
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
                domain_feat = domain_feat[:1024, :]  # [1024, hidden_size of esm]
                domain_feat_mask = domain_feat_mask[:1024]  # [1024]
            domain_feats_padded.append(domain_feat)
            domain_feat_masks_padded.append(domain_feat_mask)

        domain_feats = torch.stack(
            domain_feats_padded
        )  # [bs, seqlen=1024, hidden_size of esm]
        domain_feats = self.domain_feats_projector(
            domain_feats
        )  # [bs, seqlen=1024, hidden_size of dplm]
        domain_feat_masks = torch.stack(domain_feat_masks_padded)  # [bs, seqlen=1024]
        return domain_feats, domain_feat_masks

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
            processed_string = re.sub(r"<pad>", "", processed_string)
            processed_string = processed_string.replace(" ", "")
            cleaned_data.append(processed_string)
        return cleaned_data




    def generate_simple(
        self,
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
        generation_config: GenerationConfig,
        use_cache: bool = True,
        verbose=True,
        **kwargs,
    ):
        """
        简化版生成方法 - 仅支持batch_size=1
        当domain piece展开时清空cache,使用简单的KV cache机制

        Args:
            domain_pieces_ids: domain pieces的token ids [num_pieces, seq_len]
            domain_pieces_masks: domain pieces的attention mask [num_pieces, seq_len]
            num_domain_pieces_per_protein: 每个蛋白质的domain piece数量
            generation_config: 生成配置
            use_cache: 是否使用KV cache加速生成
        Returns:
            生成的序列字典
        """
        batch_size = len(num_domain_pieces_per_protein)
        assert batch_size == 1, "简化版仅支持batch_size=1"

        # 1. 使用统一的encoder编码domain pieces
        encoder_out, encoder_mask, pooler_output = self.forward_unified_encoder(
            domain_pieces_ids,
            domain_pieces_masks,
            num_domain_pieces_per_protein
        )

        # 2. 用于扩展词表
        phrase_out = self.unified_projector(pooler_output)

        # 3. 初始化输入
        input_ids = self.initialize_output_tokens(batch_size)  # [1, 1]

        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        max_length = generation_config.max_length if generation_config.max_length else 512
        vocab_size = self.qwen3.config.vocab_size

        # 扩展lm_head权重
        lm_head_weights = self.qwen3.lm_head.weight
        extended_lm_head_weights = torch.cat([lm_head_weights, phrase_out], dim=0)
        total_vocab_size = extended_lm_head_weights.size(0)

        # domain_pieces_ids: [num_pieces, seq_len]
        # 将domain_pieces_ids解码为token列表,用于domain piece展开
        num_pieces = domain_pieces_ids.shape[0]
        domain_pieces_tokens = []
        for i in range(num_pieces):
            # 提取非padding的token
            piece_ids = domain_pieces_ids[i]  # [seq_len]
            piece_mask = domain_pieces_masks[i]  # [seq_len]

            # 只保留有效的token (mask=1的部分)
            valid_tokens = piece_ids[piece_mask.bool()].tolist()

            # 移除特殊token (cls, eos, pad)
            valid_tokens = [
                t for t in valid_tokens
                if t not in [self.tokenizer.cls_token_id,
                            self.tokenizer.eos_token_id,
                            self.tokenizer.pad_token_id]
            ]

            domain_pieces_tokens.append(valid_tokens)

        # 初始化
        finished = False
        step = 0
        past_key_values = None

        # Domain piece索引范围
        domain_start_in_vocab = vocab_size
        domain_end_in_vocab = vocab_size + num_domain_pieces_per_protein[0].item()

        while step < max_length and not finished:
            # 计算cache_position
            if past_key_values is not None and len(past_key_values) > 0:
                # 有cache: 只处理最后一个token
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + 1,
                    device=input_ids.device
                )
                # 只取最后一个token作为输入
                current_input_ids = input_ids[:, -1:]
            else:
                # 第一次调用或cache被清空: 处理整个序列
                seq_len = input_ids.shape[1]
                cache_position = torch.arange(seq_len, device=input_ids.device)
                current_input_ids = input_ids

            # Forward
            outputs = self.qwen3(
                input_ids=current_input_ids,
                encoder_hidden_states=encoder_out,
                encoder_attention_mask=encoder_mask,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                cache_position=cache_position,
                output_hidden_states=True,
                return_dict=True,
            )

            # 更新cache
            if use_cache:
                past_key_values = outputs.past_key_values

            # 获取最后一个token的hidden state
            hidden_states = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_size]

            # 计算扩展词表的logits
            logits = hidden_states @ extended_lm_head_weights.T  # [1, total_vocab_size]

            # Mask掉不属于当前样本的domain pieces
            mask_value = -1e10
            if domain_start_in_vocab > vocab_size:
                logits[0, vocab_size:domain_start_in_vocab] = mask_value
            if domain_end_in_vocab < total_vocab_size:
                logits[0, domain_end_in_vocab:] = mask_value

            # 采样
            if generation_config.do_sample:
                temperature = generation_config.temperature if generation_config.temperature else 1.0
                logits = logits / temperature

                if generation_config.top_k and generation_config.top_k > 0:
                    top_k = min(generation_config.top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = mask_value

                if generation_config.top_p and generation_config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = mask_value

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token_id = next_token.item()

            # 检查是否结束
            if next_token_id == self.tokenizer.eos_token_id:
                finished = True
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                step += 1
                continue

            # 检查是否是domain piece
            if next_token_id >= vocab_size:
                # Domain piece展开
                relative_idx = next_token_id - vocab_size

                if 0 <= relative_idx < len(domain_pieces_tokens):
                    domain_piece_token_seq = domain_pieces_tokens[relative_idx]
                    domain_piece_tensor = torch.tensor(
                        domain_piece_token_seq,
                        device=input_ids.device
                    ).unsqueeze(0)

                    # 拼接展开的tokens
                    input_ids = torch.cat([input_ids, domain_piece_tensor], dim=1)

                    # 清空cache (关键: domain piece展开时清空cache)
                    if use_cache:
                        past_key_values = None

                    step += len(domain_piece_token_seq)
                else:
                    # 索引越界,跳过
                    step += 1
            else:
                # 普通token,直接添加
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                step += 1

            # 检查是否超过最大长度
            if input_ids.size(1) >= max_length:
                break

        # 解码序列
        tokens = input_ids[0].cpu().tolist()
        # 移除特殊tokens
        tokens = [t for t in tokens if t != self.tokenizer.cls_token_id]
        if self.tokenizer.eos_token_id in tokens:
            eos_idx = tokens.index(self.tokenizer.eos_token_id)
            tokens = tokens[:eos_idx]
        seq = self.tokenizer.decode(tokens)

        return {
            "output_seqs": self.clean_and_format_seq([seq]),
        }


    def generate(
        self,
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
        generation_config: GenerationConfig,
        use_cache: bool = True,
        verbose=True,
        **kwargs,
    ):
        """
        生成蛋白质序列，支持RAG和扩展词表，使用KV cache加速
        当预测到domain piece token时，将其展开为完整的氨基酸序列继续生成

        Args:
            domain_pieces_ids: domain pieces的token ids
            domain_pieces_masks: domain pieces的attention mask
            num_domain_pieces_per_protein: 每个蛋白质的domain piece数量
            generation_config: 生成配置
            use_cache: 是否使用KV cache加速生成
        Returns:
            生成的序列字典
        """
        return self.generate_simple(
            domain_pieces_ids, domain_pieces_masks, num_domain_pieces_per_protein,
            generation_config, use_cache, verbose, **kwargs
        )

    def generate_with_variable_length_cache(
        self,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
        domain_pieces_seqs: list[list[str]],
        generation_config: GenerationConfig,
        use_cache: bool = True,
        verbose=True,
        **kwargs,
    ):
        """
        使用VariableLengthCache的生成方法
        当domain piece被展开时,通过左padding对齐batch,不清空cache

        核心思想:
        - 样本1预测domain_piece展开为[X,Y,Z] (3个token)
        - 样本2预测普通token P (1个token)
        - 对齐: 样本1 [X,Y,Z], 样本2 [pad,pad,P]
        - Cache只追加有效token: 样本1追加3个, 样本2追加1个
        """
        from .VariableLengthCache import VariableLengthCache
        from .Qwen3.modeling_domainconditioning_qwen3 import EncoderDecoderCache

        batch_size = len(num_domains_per_protein)


        # 1. 编码domain特征（用于交叉注意力）
        encoder_out, encoder_mask = self.forward_encoder(
            domain_ids, domain_masks, num_domains_per_protein
        )

        # 2. 编码domain pieces（用于扩展词表）
        phrase_out = self.phrase_feats_projector(
            self.phrase_encoder(
                input_ids=domain_pieces_ids,
                attention_mask=domain_pieces_masks,
                return_dict=True,
            ).pooler_output
        )

        # 3. 初始化输入
        input_ids = self.initialize_output_tokens(batch_size)  # [bs, 1]

        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        max_length = generation_config.max_length if generation_config.max_length else 512
        vocab_size = self.qwen3.config.vocab_size

        # 扩展lm_head权重
        lm_head_weights = self.qwen3.lm_head.weight
        extended_lm_head_weights = torch.cat([lm_head_weights, phrase_out], dim=0)
        total_vocab_size = extended_lm_head_weights.size(0)

        # 预计算domain起始索引
        if len(num_domain_pieces_per_protein) > 1:
            domain_start_indices = torch.cumsum(
                torch.cat([
                    torch.tensor([0], device=num_domain_pieces_per_protein.device,
                                dtype=num_domain_pieces_per_protein.dtype),
                    num_domain_pieces_per_protein[:-1]
                ]),
                dim=0,
            )
        else:
            domain_start_indices = torch.tensor([0], device=num_domain_pieces_per_protein.device,
                                               dtype=num_domain_pieces_per_protein.dtype)

        # 预先tokenize domain pieces
        domain_pieces_tokens = []
        for i in range(batch_size):
            sample_tokens = []
            for seq in domain_pieces_seqs[i]:
                tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                sample_tokens.append(tokens)
            domain_pieces_tokens.append(sample_tokens)

        # 初始化
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        step = 0
        attention_mask = torch.ones_like(input_ids)

        # 使用自定义的VariableLengthCache包装在EncoderDecoderCache中
        if use_cache:
            past_key_values = EncoderDecoderCache(
                VariableLengthCache(),  # self-attention cache
                VariableLengthCache()   # cross-attention cache
            )
        else:
            past_key_values = None

        while step < max_length and not finished.all():
            # 计算cache_position
            if past_key_values is not None and len(past_key_values.self_attention_cache) > 0:
                # 有cache: 从cache长度开始
                past_seen_tokens = past_key_values.get_seq_length()
                current_seq_len = input_ids.shape[1]
                num_new_tokens = current_seq_len - past_seen_tokens

                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + num_new_tokens,
                    device=input_ids.device
                )
            else:
                # 第一次调用: cache_position = [0, 1, ..., seq_len-1]
                seq_len = input_ids.shape[1]
                cache_position = torch.arange(seq_len, device=input_ids.device)

            # Forward - 传入attention_mask以便cache能够推断有效长度
            outputs = self.qwen3(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_out,
                encoder_attention_mask=encoder_mask,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                cache_position=cache_position,
                output_hidden_states=True,
                return_dict=True,
            )

            # 获取最后一个有效位置的hidden state
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

            # 对于每个样本,取最后一个有效token的hidden state
            # 由于是左padding,最后一个有效token总是在最右边
            last_hidden_states = hidden_states[:, -1, :]  # [batch_size, hidden_size]

            # 计算扩展词表的logits
            logits = last_hidden_states @ extended_lm_head_weights.T  # [batch_size, total_vocab_size]

            # Mask掉不属于当前样本的domain pieces
            mask_value = -1e10
            for i in range(batch_size):
                if finished[i]:
                    continue

                start_idx = domain_start_indices[i].item()
                end_idx = start_idx + num_domain_pieces_per_protein[i].item()

                domain_start_in_vocab = vocab_size + start_idx
                domain_end_in_vocab = vocab_size + end_idx

                if domain_start_in_vocab > vocab_size:
                    logits[i, vocab_size:domain_start_in_vocab] = mask_value
                if domain_end_in_vocab < total_vocab_size:
                    logits[i, domain_end_in_vocab:] = mask_value

            # 采样
            if generation_config.do_sample:
                temperature = generation_config.temperature if generation_config.temperature else 1.0
                logits = logits / temperature

                if generation_config.top_k and generation_config.top_k > 0:
                    top_k = min(generation_config.top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = mask_value

                if generation_config.top_p and generation_config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = mask_value

                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            # 处理domain piece展开,计算每个样本新增的token数
            new_token_sequences = []
            max_new_length = 1

            for i in range(batch_size):
                if finished[i]:
                    new_token_sequences.append([])
                    continue

                next_token = next_tokens[i].item()

                if next_token == self.tokenizer.eos_token_id:
                    finished[i] = True
                    new_token_sequences.append([next_token])
                elif next_token >= vocab_size:
                    # Domain piece展开
                    domain_piece_idx = next_token - vocab_size
                    sample_start_idx = domain_start_indices[i].item()
                    relative_idx = domain_piece_idx - sample_start_idx

                    if 0 <= relative_idx < len(domain_pieces_tokens[i]):
                        domain_piece_token_seq = domain_pieces_tokens[i][relative_idx]
                        new_token_sequences.append(domain_piece_token_seq)
                        max_new_length = max(max_new_length, len(domain_piece_token_seq))
                    else:
                        new_token_sequences.append([])
                else:
                    # 普通token
                    new_token_sequences.append([next_token])

            # 左padding对齐所有样本
            aligned_sequences = []

            for i in range(batch_size):
                new_tokens = new_token_sequences[i]
                valid_len = len(new_tokens)

                if valid_len < max_new_length:
                    # 左padding
                    padding = [self.tokenizer.pad_token_id] * (max_new_length - valid_len)
                    aligned_tokens = padding + new_tokens
                else:
                    aligned_tokens = new_tokens

                aligned_sequences.append(aligned_tokens)

            # 拼接到input_ids
            new_input_ids = []
            for i in range(batch_size):
                new_tokens = torch.tensor(aligned_sequences[i], device=input_ids.device)
                updated_seq = torch.cat([input_ids[i], new_tokens])
                new_input_ids.append(updated_seq)

            # 左padding整个batch
            max_len = max(seq.size(0) for seq in new_input_ids)
            padded_input_ids = []
            padded_attention_mask = []

            for i, seq in enumerate(new_input_ids):
                seq_len = seq.size(0)
                if seq_len < max_len:
                    pad_len = max_len - seq_len
                    seq = torch.cat([
                        torch.full((pad_len,), self.tokenizer.pad_token_id, device=seq.device),
                        seq
                    ])

                # attention mask: 1表示有效token, 0表示padding
                # 计算有效长度: 从右边数,直到遇到第一个非pad token
                mask = (seq != self.tokenizer.pad_token_id).long()

                padded_input_ids.append(seq)
                padded_attention_mask.append(mask)

            input_ids = torch.stack(padded_input_ids)
            attention_mask = torch.stack(padded_attention_mask)

            step += max_new_length

            if input_ids.size(1) >= max_length:
                break

        # 解码序列
        sequences = []
        for i in range(batch_size):
            tokens = input_ids[i].cpu().tolist()
            # 移除padding和特殊tokens
            tokens = [t for t in tokens if t != self.tokenizer.pad_token_id and t != self.tokenizer.cls_token_id]
            if self.tokenizer.eos_token_id in tokens:
                eos_idx = tokens.index(self.tokenizer.eos_token_id)
                tokens = tokens[:eos_idx]
            seq = self.tokenizer.decode(tokens)
            sequences.append(seq)

        return {
            "output_seqs": self.clean_and_format_seq(sequences),
        }
